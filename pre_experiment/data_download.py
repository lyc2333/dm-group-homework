import kagglehub

# Download latest version
path = kagglehub.dataset_download("vikawenzel/scraped-steam-data-games-reviews-charts")

print("Path to dataset files:", path)
