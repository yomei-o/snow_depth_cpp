# docs/ をローカルで配信し、playwright でサンプルを順に押して点数を取る＋画面のスクショ。
import sys, threading, http.server, functools, json
from playwright.sync_api import sync_playwright
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
root = sys.argv[1]; shot = sys.argv[2]
H = functools.partial(http.server.SimpleHTTPRequestHandler, directory=root)
H.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8765), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 760, "height": 1100})
    pg.on("console", lambda m: print("console:", m.text))
    pg.goto("http://127.0.0.1:8765/index.html")
    pg.wait_for_function("document.getElementById('status').textContent.startsWith('準備')", timeout=180000)
    n = pg.locator("#samples img").count()
    for i in range(n):
        pg.locator("#samples img").nth(i).click()
        pg.wait_for_function("document.getElementById('status').textContent.startsWith('判定しました')", timeout=60000)
        print(pg.locator("#samples img").nth(i).get_attribute("src"), '路面', pg.text_content('#roadLabel'), pg.text_content('#roadCm'), 'まわり', pg.text_content('#aroundLabel'), pg.text_content('#comment'), pg.text_content("#status"))
        pg.evaluate("document.getElementById('status').textContent=''")
    pg.screenshot(path=shot, full_page=True)
    b.close()
srv.shutdown()
