#!/usr/bin/python

# MIT
# Copyright (c) 2015 Marc Juarez <marc.juarez@kuleuven.be>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This module parses pcap files for website fingerprinting studies.

# Note that if pcaps are small, multiprocessing does not have a great impact as
# the IO to disk becomes the bottelneck of the program.

from __future__ import print_function
from functools import partial
from subprocess import call
import contextlib
import dpkt
from sys import stderr, stdout
import struct
import re
import argparse
import shutil
from time import time
import multiprocessing as mp
from itertools import izip, repeat
from os import stat, walk, remove, getpid
from os.path import isdir, isfile, join, split

IN = -1
OUT = 1
CELL_LENGTH = 512
P1, P2 = 45, 40
FIELDS = ["packet_index", "timestamp", "length", "direction", "num_cells", "ack"]
PRIVATE_IP_REGEXP = re.compile("(^127\.0\.0\.1)|"
                               "(^10\.)|"
                               "(^172\.1[6-9]\.)|"
                               "(^172\.2[0-9]\.)|"
                               "(^172\.3[0-1]\.)|"
                               "(^192\.168\.)")

__decoder = {dpkt.pcap.DLT_LOOP: dpkt.loopback.Loopback,
             dpkt.pcap.DLT_NULL: dpkt.loopback.Loopback,
             dpkt.pcap.DLT_EN10MB: dpkt.ethernet.Ethernet,
             dpkt.pcap.DLT_LINUX_SLL: dpkt.sll.SLL}


def memodict(f):
    """Memoization decorator for a function taking a single argument."""
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret
    return memodict().__getitem__


@contextlib.contextmanager
def wopen(filename=None):
    if filename and filename != '-':
        fh = open(filename, 'wb')
    else:
        fh = stdout
    try:
        yield fh
    finally:
        if fh is not stdout:
            fh.close()


@memodict
def to_ip(ip_bytes):
    """Return 4 bytes to a string with IP format (o1.o2.o3.o4)."""
    return ".".join(map(str, struct.unpack("!BBBB", ip_bytes)))


def get_num_cells(record_length):
    """Return number of cells in TLS record."""
    # Round down to closest multiple of 512
    round_cell = record_length - (record_length % CELL_LENGTH)
    # Divide by cell length
    num_cells = round_cell / CELL_LENGTH
    return num_cells


def gen_parse_pcap(pcap_reader, fields=FIELDS, sendme=False, noack=False):
    """Generator that yields attributes of packets in a pcap."""
    sendme_count = 0
    #seen_seq = set()
    seen_seq = dict()
    # Parse pcap
    decode = __decoder[pcap_reader.datalink()]
    for i, (timestamp, buf) in enumerate(pcap_reader):
        parsed = {"packet_index": i, "timestamp": timestamp, "direction": IN}
        try:
            # Disassemble packet
            pkt = decode(buf)
            if isinstance(pkt, str):
                raise Exception("Packet number %s could not be parsed."
                                "Maybe the layer 2 protocol is not Ethert." % i)
            parsed["length"] = len(pkt)
            ip = pkt.data
            if not ip.p == dpkt.ip.IP_PROTO_TCP:
                print("WARNING\tNon TCP packet found.", file=stderr)
                continue
            tcp = ip.data
            record_length = len(tcp.data)
            # Ignore ACKs
            if noack and record_length == 0:
                continue
            # Find if pkt is ACK with no payload
            parsed["ack"] = int(record_length == 0 and (tcp.flags & dpkt.tcp.TH_ACK) != 0)
            # Ignore retransmissions
            #if tcp.seq in seen_seq:
            #   continue
            #seen_seq.add(tcp.seq)
            if tcp.seq in seen_seq.keys() and seen_seq[tcp.seq] == record_length:
                continue
            seen_seq[tcp.seq] = record_length
            # Find direction
            if PRIVATE_IP_REGEXP.search(to_ip(ip.src)):
                parsed["direction"] = OUT
            # Find number of cells in packet
            parsed['num_cells'] = get_num_cells(record_length)
            # Remove SENDMEs
            if sendme:
                for _ in xrange(parsed['num_cells']):
                    if parsed["direction"] < 0:
                        sendme_count += 1
                    elif sendme_count >= P1:
                        sendme_count -= P2
                        continue
            yield parsed
        except Exception as e:
            print("EXCEPTION\tWhile parsing packet", e, file=stderr)


def tostr(x):
    if isinstance(x, float):
        return "%.6f" % x
    return str(x)

def join_sep(field_list, sep=';'):
    return '{}\n'.format(sep.join(map(tostr, field_list)))


def parse_file(fields, sep, sendme, noack, in_fpath, output_fpath, pcapng=False,
               **kwargs):
    """Parse pcap file at `in_fpath`."""
    print("INFO\tParsing pcap:", in_fpath, file=stderr)

    try:
        with open(in_fpath, "rb") as fin:
            if stat(in_fpath).st_size == 0:
                print("WARNING\tEmpty pcap:", in_fpath, file=stderr)
                return
            pcap_reader = dpkt.pcap.Reader(fin)
            output = join_sep(fields, sep)
            for parsed in gen_parse_pcap(pcap_reader, fields, sendme, noack):
                output += join_sep([parsed[v] for v in fields], sep)
    except ValueError as e1:
        print("EXCEPTION\t%s. Trying to convert from pcapng to pcap." % e1,
              file=stderr)
        if e1.message == 'invalid tcpdump header' and not pcapng:
            mypid = getpid()
            ts = str(int(time() * 1000000))
            tmp_pcapng_fname = ".tmp{0}_{1}.pcapng".format(mypid, ts)
            tmp_pcap_fname = ".tmp{0}_{1}.pcap".format(mypid, ts)
            try:
                shutil.copyfile(in_fpath, tmp_pcapng_fname)
                call(["editcap", "-F", "libpcap", "-T", "ether",
                      tmp_pcapng_fname, tmp_pcap_fname])
                parse_file(fields, sep, sendme, noack, tmp_pcap_fname, output_fpath,
                           pcapng=True, **kwargs)
            except KeyboardInterrupt:
                pass
            except Exception as e2:
                print("EXCEPTION\tWhile converting from pcapng to pcap: %s"
                      % e2, file=stderr)
                print("INFO\tConversion requires: editcap command-line tool.",
                      file=stderr)
            finally:
                if isfile(tmp_pcapng_fname):
                    remove(tmp_pcapng_fname)
                if isfile(tmp_pcap_fname):
                    remove(tmp_pcap_fname)
    except Exception as e3:
        print("EXCEPTION\tPcap %s could not ve parsed: %s" % (in_fpath, e3),
              file=stderr)
    else:
        with wopen(output_fpath) as fout:
            fout.write(output)



def ispcap(fpath):
    return isfile(fpath) and fpath.endswith('.pcap')


def gen_find_pcaps(in_fpath):
    """Generator that yields pcaps in `in_path` directory."""
    for root, _, files in walk(in_fpath):
        if not files:
            continue
        only_pcaps = [join(root, f) for f in files if ispcap(join(root, f))]
        for pcap in only_pcaps:
            yield pcap



def gen_get_csvs(in_fpath, out_fpath):
    """Generator that yields csv filenames."""
    for pcap in gen_find_pcaps(in_fpath):
        pcap_base, pcap_fname = split(pcap)
        csv_fname = "{}.csv".format(pcap_fname[:-5])
        if out_fpath:
            yield join(out_fpath, csv_fname)
        else:
            yield join(pcap_base, csv_fname)



def star(args):
    return args[0](*args[1:])


def parse_dir(fields, sep, sendme, noack, in_fpath, output_fpath, num_procs):
    """Parse pcaps from `in_fpath` and write them to output directory."""
    proc_pool = mp.Pool(num_procs)
    print("INFO\tLaunching %s processes." % num_procs, file=stderr)
    gpcaps = gen_find_pcaps(in_fpath)
    gcsvs = gen_get_csvs(in_fpath, output_fpath)
    partialpf = partial(parse_file, fields, sep, sendme, noack)
    tasks = izip(repeat(partialpf), gpcaps, gcsvs)
    proc_pool.map(star, tasks)


def main(args):
    if isfile(args.in_fpath):
        parse_file(**vars(args))
    elif isdir(args.in_fpath):
        parse_dir(**vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse a pcap file.')

    parser.add_argument('--in-fpath', '-i',
                        required=True,
                        type=str,
                        help='''path to pcap file or directory from where
                                to read.''')

    parser.add_argument('--output-fpath', '-o',
                        type=str,
                        default=None,
                        help='path to the output file (default=stdout).')

    parser.add_argument('--fields', '-f',
                        nargs='*',
                        choices=FIELDS,
                        default=FIELDS,
                        help='list of fields in output (default=all).')

    parser.add_argument('--sendme', '-m',
                        action='store_true',
                        default=False,
                        help='''whether we try to Tor SENDME cells
                                (default=False)''')

    parser.add_argument('--noack', '-a',
                        action='store_true',
                        default=False,
                        help='''whether we ignore the TCP acknowlegements
                                (default=False)''')
                                
    parser.add_argument('--num_procs', '-p',
                        type=int,
                        default=mp.cpu_count(),
                        help='''number of processes to spawn for pasing.
                                Only for parsing directories
                                (default=available in host).''')

    parser.add_argument('--sep', '-s',
                        type=str,
                        default=';',
                        help='field separator of the output (default=;)')

    args = parser.parse_args()
    main(args)
