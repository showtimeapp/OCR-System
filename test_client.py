"""Test client — send PDF, download results"""
import requests, sys, time

API = "http://localhost:8000"

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
            print(f'\nDownload:')
            print(f'  JSON: {API}{data["download_json"]}')
            print(f'  MD:   {API}{data["download_md"]}')
            print(f'  ZIP:  {API}{data["download_zip"]}')

            # Save locally
            job_id = data['job_id']
            for endpoint, filename in [
                (data['download_json'], f'{job_id}_extraction.json'),
                (data['download_md'], f'{job_id}_full.md'),
            ]:
                r = requests.get(f'{API}{endpoint}')
                with open(filename, 'wb') as out:
                    out.write(r.content)
                print(f'  Saved: {filename}')

        else:
            # Async — poll for completion
            resp = requests.post(f'{API}/extract/async', files=files)
            data = resp.json()
            job_id = data['job_id']
            print(f'Job started: {job_id}')

            while True:
                status = requests.get(f'{API}/status/{job_id}').json()
                if status['status'] == 'done':
                    print(f'\nDone!')
                    print(f'  JSON: {API}{status["download_json"]}')
                    print(f'  MD:   {API}{status["download_md"]}')
                    break
                elif status['status'] == 'error':
                    print(f'Error: {status["error"]}')
                    break
                else:
                    print(f'  Processing...', end='\r')
                    time.sleep(5)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python test_client.py <pdf_path>')
    else:
        extract(sys.argv[1])
