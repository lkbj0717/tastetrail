"""Read-only UI QA. Optionally render a saved fictional live response; no POST."""
import argparse
import json
from pathlib import Path

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('WebKit2', '4.0')
from gi.repository import Gtk, WebKit2, GLib

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8773/')
parser.add_argument('--response', type=Path)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
log = {'browser': 'WebKitGTK', 'checks': [], 'paid_api_calls': 0,
       'saved_live_response_rendered': bool(args.response)}
window = Gtk.Window()
window.set_default_size(1400, 1050)
view = WebKit2.WebView()
window.add(view)
window.show_all()
finished = False

def finish(error=None):
    global finished
    if finished:
        return False
    finished = True
    log['passed'] = error is None
    if error:
        log['error'] = str(error)
    (args.out / 'browser_qa.json').write_text(json.dumps(log, indent=2))
    print(json.dumps(log), flush=True)
    Gtk.main_quit()
    return False

def js(code, callback):
    def done(web, result, *_):
        try:
            callback(web.evaluate_javascript_finish(result).to_string())
        except Exception as exc:
            finish(exc)
    view.evaluate_javascript(code, -1, None, None, None, done, None)

def shot(name, callback):
    def done(web, result, *_):
        try:
            web.get_snapshot_finish(result).write_to_png(str(args.out / name))
            callback()
        except Exception as exc:
            finish(exc)
    view.get_snapshot(WebKit2.SnapshotRegion.FULL_DOCUMENT,
                      WebKit2.SnapshotOptions.NONE, None, done, None)

def initial(raw):
    data = json.loads(raw)
    assert data['history'] == 8 and data['candidates'] == 2, data
    assert data['catalog'] == 473 and not data['overflow'], data
    log['checks'].append(data)
    shot('demo_desktop.png', invalid)

def invalid():
    js("document.getElementById('likes-0').value='unsupported film';"
       "document.getElementById('run').click();"
       "document.getElementById('status').textContent", invalid_done)

def invalid_done(raw):
    assert 'supported film' in raw, raw
    log['checks'].append({'unsupported_film_rejected_locally': True})
    if args.response:
        response = json.loads(args.response.read_text())
        js('fillForm(example);activeBody=readForm();render(' +
           json.dumps(response, ensure_ascii=False) + ");status('Saved live comparison rendered');" +
           "JSON.stringify({cards:document.querySelectorAll('.result-card').length,"
           "trace:document.querySelectorAll('.trace li').length,"
           "overflow:document.documentElement.scrollWidth>innerWidth})", result_done)
    else:
        mobile()

def result_done(raw):
    data = json.loads(raw)
    assert data['cards'] == 3 and data['trace'] > 0 and not data['overflow'], data
    log['checks'].append(data)
    shot('demo_live_result.png', mobile)

def mobile():
    window.resize(390, 844)
    GLib.timeout_add(900, lambda: js(
        "JSON.stringify({width:innerWidth,overflow:document.documentElement.scrollWidth>innerWidth})",
        mobile_done) or False)

def mobile_done(raw):
    data = json.loads(raw)
    assert data['width'] <= 400 and not data['overflow'], data
    log['checks'].append(data)
    shot('demo_mobile.png', lambda: finish())

started = False
def loaded(web, event):
    global started
    if event == WebKit2.LoadEvent.FINISHED and not started:
        started = True
        GLib.timeout_add(1400, lambda: js(
            "JSON.stringify((()=>{const b=readForm();return {history:b.likes.length+b.dislikes.length,"
            "candidates:b.candidates.length,catalog:filmLookup.size,"
            "overflow:document.documentElement.scrollWidth>innerWidth}})())", initial) or False)

view.connect('load-changed', loaded)
view.load_uri(args.url)
GLib.timeout_add_seconds(35, lambda: finish('Browser QA timed out'))
Gtk.main()
raise SystemExit(0 if log.get('passed') else 1)
