duplicates = [["ablogica.com", "queryrouter.com"], ["sogou.com", "soso.com"], ["sh.st", "viid.me"], ["reddit.com", "redd.it"], ["pinterest.com", "pinimg.com"], ["blogger.com", "blogspot.com"], ["facebook.com", "fbcdn.net"], ["google.com", "alxsite.com", "clotraiam.website", "prestoris.com", "googlevideo.com"], ["tumblr.com", "umblr.com"], ["mercadolivre.com.br", "mercadolibre.com.ar"]]
empty = ["s34me.com", "trackmedia101.com", "ab4hr.com", "adplxmd.com", "list-manage.com", "clicksgear.com", "terraclicks.com", "watermelon-shake.com", "go2cloud.org", "thewhizproducts.com"]
almost_empty = ["onclkds.com", "thewhizmarketing.com"]
unknown_error = ["etsy.com", "craigslist.org"]
similar = [["salesforce.com", "force.com"], ["theladbible.com", "thesportbible.com"]]

# 211 valid targets from the first 300 with at least 90 vists per run (at least 4500 visits in total)
targets_runs_0_49_tr_90 = ['irctc.co.in', 'xnxx.com', 'savefrom.net', 'trello.com', 'mailchimp.com', 'porn555.com',
                     'themeforest.net', 'daikynguyenvn.com', 'onedio.com', 'whatsapp.com', 'spotscenered.info',
                     'hclips.com', 'office.com', 'stackoverflow.com', 'subscene.com', 'goo.ne.jp', 'pandora.com',
                     'researchgate.net', 'intuit.com', 'businessinsider.com', 'aol.com', 'wikia.com', 'txxx.com',
                     'bongacams.com', 'daum.net', 'imgur.com', 'tripadvisor.com', 'netflix.com', 'jd.com',
                     'dictionary.com', 'steampowered.com', 'slideshare.net', 'alibaba.com', 'diply.com',
                     'weather.com', 'ettoday.net', 'xhamster.com', 'tumblr.com', 'webtretho.com', 'wix.com',
                     'flipkart.com', 'onclkds.com', 'xvideos.com', 'skype.com', 'uptodown.com', 'softonic.com',
                     'allegro.pl', 'blastingnews.com', 'chaturbate.com', 'ebay.com', 'nih.gov', 'gfycat.com',
                     'nametests.com', 'openload.co', 'buzzfeed.com', 'force.com', 'booking.com', 'quora.com',
                     'orange.fr', 'reimageplus.com', 'rolloid.net', 'nicovideo.jp', 'ask.com', 'godaddy.com',
                     'theladbible.com', 'ntd.tv', 'livedoor.jp', 'rambler.ru', 'evernote.com', 'conservativetribune.com',
                     'ok.ru', '9gag.com', 'wellsfargo.com', 'theguardian.com', 'espncricinfo.com', 'go.com',
                     'seasonvar.ru', 'ltn.com.tw', 'twitter.com', 'ameblo.jp', 'gmx.net', 'youtube-mp3.org',
                     'amazon.com', 'battle.net', 'vk.com', 'twitch.tv', 'dingit.tv', 'goodreads.com', 'wordpress.com',
                     'cnn.com', 'yelp.com', 'yahoo.com', 'tokopedia.com', 'popcash.net', 'weebly.com', 'imdb.com',
                     'oracle.com', 'stackexchange.com', 'ikea.com', 'hotstar.com', 'dropbox.com', 'kaskus.co.id',
                     'rt.com', 'appspot.com', 'torrentz2.eu', 'reddit.com', 'wordreference.com', 'coccoc.com',
                     'mercadolivre.com.br', 'aliexpress.com', 'naver.com', 'feedly.com', 'deviantart.com',
                     'dailymotion.com', 'wetransfer.com', 'scribd.com', 'detik.com', 'wikipedia.org', 'livejournal.com',
                     'lifebuzz.com', 'freepik.com', 'web.de', 'myway.com', 'vimeo.com', 'instructure.com', 'bbc.co.uk',
                     'kompas.com', 'nytimes.com', 'bing.com', 'microsoft.com', 'sciencedirect.com', 'w3schools.com',
                     'pinterest.com', 'shutterstock.com', 'salesforce.com', 'rakuten.co.jp', 'avito.ru', 'fc2.com',
                     'upornia.com', 'sharepoint.com', 'roblox.com', 'amazonaws.com', 'hotmovs.com', 'speedtest.net',
                     'zippyshare.com', 'blogger.com', 'discordapp.com', 'popads.net', 'github.com', 'facebook.com',
                     'kakaku.com', 'google.com', 'bukalapak.com', 'sabah.com.tr', 'abs-cbn.com', 'mozilla.org',
                     'providr.com', 'washingtonpost.com', 'apple.com', 'adobe.com', 'doubleclick.net', 'extratorrent.cc',
                     'hatena.ne.jp', 'espn.com', 'linkedin.com', 'tribunnews.com', 'yandex.ru', 'leboncoin.fr',
                     'youtube.com', 'breitbart.com', 'dailymail.co.uk', 'spotify.com', 'varzesh3.com', 'outbrain.com',
                     'messenger.com', 'archive.org', 'liputan6.com', 'mail.ru', 'exoclick.com', 'redtube.com', 'gamepedia.com',
                     'microsoftonline.com', 't.co', 'thesaurus.com', 'mediafire.com', 'tistory.com', 'soundcloud.com', 'adf.ly',
                     'askcom.me', 'blackboard.com', 'livejasmin.com', 'digikala.com', 'vice.com', 'sourceforge.net', 'chase.com',
                     'ouo.io', 'hola.com', 'thewhizmarketing.com', 'steamcommunity.com', 'slack.com', 'douyu.com', 'yts.ag',
                     'youm7.com', 'ebay-kleinanzeigen.de', 'wikihow.com', 'wikimedia.org', 'doublepimp.com', 'msn.com',
                     'onlinesbi.com', 'youporn.com', 'wittyfeed.com']

# 100 targets with 200 vists in runs 2 and 3
targets_runs_2_3_tr_100 = ['onedio.com','spotscenered.info', 'hclips.com', 'office.com', 'stackoverflow.com', 'researchgate.net', 'intuit.com',
                            'businessinsider.com', 'aol.com', 'wikia.com', 'netflix.com', 'dictionary.com', 'steampowered.com',
                            'weather.com', 'xhamster.com', 'tumblr.com', 'xvideos.com', 'skype.com', 'allegro.pl', 'blastingnews.com',
                            'ebay.com', 'nih.gov', 'openload.co', 'booking.com', 'quora.com', 'nicovideo.jp', 'godaddy.com', 'ntd.tv',
                            'conservativetribune.com', '9gag.com', 'theguardian.com', 'espncricinfo.com', 'go.com', 'ltn.com.tw',
                            'twitter.com', 'amazon.com', 'battle.net', 'goodreads.com', 'yelp.com', 'tokopedia.com', 'imdb.com',
                            'oracle.com', 'stackexchange.com', 'rt.com', 'torrentz2.eu', 'aliexpress.com', 'naver.com', 'feedly.com',
                            'deviantart.com', 'dailymotion.com', 'scribd.com', 'detik.com', 'wikipedia.org', 'livejournal.com',
                            'instructure.com', 'kompas.com', 'pinterest.com', 'roblox.com', 'amazonaws.com', 'hotmovs.com', 'zippyshare.com',
                            'popads.net', 'github.com', 'facebook.com', 'google.com', 'bukalapak.com', 'abs-cbn.com', 'mozilla.org',
                            'rutracker.org', 'apple.com', 'adobe.com', 'doubleclick.net', 'extratorrent.cc', 'tribunnews.com', 'yandex.ru',
                            'leboncoin.fr', 'breitbart.com', 'dailymail.co.uk', 'spotify.com', 'outbrain.com', 'archive.org', 'liputan6.com',
                            'exoclick.com', 'gamepedia.com', 'microsoftonline.com', 't.co', 'thesaurus.com', 'tistory.com', 'soundcloud.com',
                            'adf.ly', 'askcom.me', 'livejasmin.com', 'digikala.com', 'sourceforge.net', 'ouo.io', 'hola.com', 'steamcommunity.com',
                            'doublepimp.com', 'msn.com', 'wittyfeed.com']