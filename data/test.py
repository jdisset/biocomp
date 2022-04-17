import webview
url = "http://localhost:8888/lab?token=9640eaf1292a39997c22dc42b6972d7b52f6567e1aec2457"
title = 'Python console'

webview.create_window(title, url=url, html='', js_api=None, width=800, height=600, \
                      x=None, y=None, resizable=True, fullscreen=False, \
                      min_size=(200, 100), hidden=False, frameless=True, \
                      minimized=False, on_top=False, confirm_close=False, \
                      background_color='#FFF', text_select=False)
webview.start()
