"""Test client — send PDF, download results"""
import requests, sys, time

API = "http://34.14.211.163"

def extract(pdf_path):
    print(f'Uploading {pdf_path}...')
    with open(pdf_path, 'rb') as f:
        t0 = time.time()
        resp = requests.post(f'{API}/extract', files={'file': (pdf_path.split('/')[-1], f)}, timeout=3600)
        dt = time.time() - t0

    if resp.status_code != 200:
        print(f'Error: {resp.text}')
        return

    data = resp.json()
    print(f'\nDone in {dt:.1f}s!')
    print(f'  Pages: {data["pages_processed"]}')
    print(f'  Charts: {data["charts_found"]}')
    print(f'  Speed: {data["avg_sec_per_page"]}s/page')

    for ep, fn in [(data['download_json'], 'extraction.json'), (data['download_md'], 'full.md')]:
        r = requests.get(f'{API}{ep}')
        with open(fn, 'wb') as f: f.write(r.content)
        print(f'  Saved: {fn}')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python test_client.py <pdf_path>')
    else:
        extract(sys.argv[1])
