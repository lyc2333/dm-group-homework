import time
import pandas as pd
import requests
import re


def get_steam_data(app_id):
    url = "https://store.steampowered.com/api/appdetails"

    while True:
        try:
            resp = requests.get(url=url, params={"appids": app_id, "l": "English"})
            if resp.status_code == 200:
                steam_data = resp.json()[str(app_id)]['data']
                return steam_data
        except Exception as e:
            print(e)
            time.sleep(0.2)


def get_steamspy_data(app_id):
    url = "https://steamspy.com/api.php"

    while True:
        try:
            resp = requests.get(url=url, params={"request": "appdetails", "appid": app_id})
            if resp.status_code == 200:
                steamspy_data = resp.json()
                return steamspy_data
        except Exception as e:
            print(e)
            time.sleep(0.2)


def unify_data(steam_data, steamspy_data):
    """
    统一格式
    """
    languages_part = steam_data.get('supported_languages', '')  # 'English<strong>*</strong>, French<strong>*</strong>, Italian<strong>*</strong>, German<strong>*</strong>, Spanish - Spain<strong>*</strong>, Arabic, Portuguese - Brazil, Polish, Traditional Chinese, Japanese<strong>*</strong>, Korean, Russian, Simplified Chinese, Spanish - Latin America<br><strong>*</strong>languages with full audio support'

    # 拆分语言部分和注释部分
    languages_part = languages_part.split('<br>')[0]

    # 用正则解析语言
    matches = re.findall(r'([^,<]+?(?:<strong>\*<\/strong>)?)(?:,|$)', languages_part)

    # 去除首尾空格并分类
    all_languages = []
    audio_languages = []

    for match in matches:
        lang = match.strip()
        if '<strong>*</strong>' in lang:
            lang = lang.replace('<strong>*</strong>', '')
            audio_languages.append(lang)
        all_languages.append(lang)
    price = int(steamspy_data['price'])/100 if ('price' in steamspy_data and steamspy_data['price'] is not None and steamspy_data['price'] != '') else 0.0
    res = {
        'name': steam_data['name'],
        'release_date': steam_data['release_date']['date'],
        'required_age': 0,  # 有好几个rating，不清楚用的哪一个，而且最后模型也没用，放这里只是为了字段统一
        'price': price,
        'dlc_count': len(steam_data['dlc']) if 'dlc' in steam_data else 0,
        'detailed_description': steam_data['detailed_description'],
        'about_the_game': steam_data['about_the_game'],
        'short_description': steam_data['short_description'],
        'reviews': steam_data['reviews'] if 'reviews' in steam_data else 0,
        'header_image': steam_data['header_image'],
        'website': steam_data['website'],
        'support_url': '',
        'support_email': '',
        'windows': steam_data['platforms']['windows'],
        'mac': steam_data['platforms']['mac'],
        'linux': steam_data['platforms']['linux'],
        'metacritic_score': steam_data['metacritic']['score'] if 'metacritic' in steam_data else 0,
        'metacritic_url': steam_data['metacritic']['url'] if 'metacritic' in steam_data else "",
        'achievements': len(steam_data['achievements']) if 'achievements' in steam_data else 0,
        'recommendations': steam_data['recommendations']['total'] if 'recommendations' in steam_data else 0,
        'notes': '',
        'supported_languages': all_languages,
        'full_audio_languages': audio_languages,
        'packages': steam_data['packages']if 'packages' in steam_data else [],
        'developers': steam_data['developers'][0] if 'developers' in steam_data else '',
        'publishers': steam_data['publishers'][0] if 'publishers' in steam_data else '',
        'categories': [item['description'] for item in steam_data['categories']] if 'categories' in steam_data else [],
        'genres': [item['description'] for item in steam_data['genres']] if 'genres' in steam_data else [],

        'screenshots': [item['path_full'] for item in steam_data['screenshots']] if 'screenshots' in steam_data else [],
        'movies': [item['mp4']['max'] for item in steam_data['movies']] if 'movies' in steam_data else [],
        'user_score': 0,
        'score_rank': "",
        'positive': steamspy_data['positive'],
        'negative': steamspy_data['negative'],
        'estimated_owners': steamspy_data['owners'].replace("..", "-").replace(",", ""),
        'average_playtime_forever': steamspy_data['average_forever'],
        'average_playtime_2weeks': steamspy_data['average_2weeks'],
        'median_playtime_forever': steamspy_data['median_forever'],
        'median_playtime_2weeks': steamspy_data['median_2weeks'],
        'discount': steamspy_data['discount'] if steamspy_data['discount']!= "" else "0",
        'peak_ccu': steamspy_data['ccu'],
        'tags': steamspy_data['tags'] if 'tags' in steamspy_data else [],
        'app_id': steamspy_data['appid'],
    }
    return res


def add_extra_feature(X):
    X['plat_support'] = X[['windows', 'mac', 'linux']].sum(axis=1)
    X['release_year'] = pd.to_datetime(X['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
    X['n_languages'] = X['supported_languages'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    X['n_categories'] = X['categories'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    X['n_genres'] = X['genres'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    X['n_tags'] = X['tags'].apply(lambda x: len(x) if isinstance(x, dict) else 0)
    X['is_family_sharing'] = X['categories'].apply(lambda x: 'Family Sharing' in x if isinstance(x, list) else False)
    X['owners_lower'] = X['estimated_owners'].str.extract(r'(\d+)').astype(float)
    X['owners_upper'] = X['estimated_owners'].str.extract(r'-\s*(\d+)').astype(float)
    X['owners_mid'] = (X['owners_lower'] + X['owners_upper']) / 2

    return X
