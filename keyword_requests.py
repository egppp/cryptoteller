
import multiprocessing
import requests
import json
from datetime import datetime, timedelta
import time

base_url = "https://api.data365.co/v1.1/twitter/search/post/update"
get_url = "https://api.data365.co/v1.1/twitter/search/post/posts"
access_token = 'ZXlKMGVYQWlPaUpLVjFRaUxDSmhiR2NpT2lKSVV6STFOaUo5LmV5SnpkV0lpT2lKRmJHbDZZV0psZEdoSGFYSmhiR1J2SWl3aWFXRjBJam94TmpnMk1EY3pNalU0TGpJM09UWXlOSDAuVDJsSEp6T0IyRjRyalNEcXNGdWRXV3hqUF80VjUtLW9iYzdNM21GOTZOZw=='


def make_requests(keyword):
    start_date = datetime.strptime("2022-07-01", "%Y-%m-%d")
    end_date = datetime.strptime("2023-01-01", "%Y-%m-%d")

    keyword_data = []
    current_date = start_date

    while current_date < end_date:
        from_date = current_date.strftime("%Y-%m-%d")
        to_date = (current_date + timedelta(days=1)).strftime("%Y-%m-%d")

        params = {
            "keywords": keyword,
            "search_type": "top",
            "max_posts": 30,
            "request": "crypto OR cryptocurrencies OR cryptocurrency",
            "lang": "en",
            "from_date": from_date,
            "to_date": to_date,
            "access_token": access_token
        }

        post_response = requests.post(base_url, params=params)

        print(post_response.text)
        print("Status code:", post_response.status_code)

        if post_response.status_code == 202:
            print("POST request successful")

            post_data = post_response.json()
            time.sleep(130)

            get_params = {
                "lang": "en",
                "keywords": keyword,
                "search_type": "latest",
                "request": "crypto OR cryptocurrencies OR cryptocurrency",
                "from_date": from_date,
                "to_date": to_date,
                "access_token": access_token 
            }

            get_response = requests.get(get_url, params=get_params)

            if get_response.status_code == 200:
                print("GET request successful")
                get_data = get_response.json()

                keyword_data.append({
                    "from_date": from_date,
                    "to_date": to_date,
                    "data": get_data
                })

            else:
                print(f"GET request failed with status code: {get_response.status_code}")
                # Handle GET request error

        else:
            print(f"POST request failed with status code: {post_response.status_code}")
            # Handle POST request error

        current_date += timedelta(days=1)

    file_name = f"{keyword}.json"
    with open(file_name, "w") as file:
        json.dump(keyword_data, file)
    print(f"Saved response for {keyword} as {file_name}")

if __name__ == "__main__":
    keywords_list = ["bitcoin OR btc", "ethereum OR eth", "binance OR bnb", "ripple OR xrp", "cardano OR ada"]
    pool = multiprocessing.Pool(processes=5)
    results = [pool.apply_async(make_requests, args=(keyword,)) for keyword in keywords_list]
    pool.close()
    pool.join()

    for result in results:
        result.get()
