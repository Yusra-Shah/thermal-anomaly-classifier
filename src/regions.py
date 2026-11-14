"""The regions of the archive files in data/raw/ and the file that belongs to each.

The first three are the original regions (Pakistan, Indonesia and the United States, each file a country
download). The others are country downloads of the same kind and the same window, January to June 2025.
"""
FILE_REGION = {
    "fire_archive_SV-C2_812937.csv": "South Asia",
    "fire_archive_SV-C2_812939.csv": "Southeast Asia",
    "fire_archive_SV-C2_812940.csv": "North and Central America",
    "fire_archive_SV-C2_847281.csv": "Brazil",
    "fire_archive_SV-C2_847293.csv": "Australia",
    "fire_archive_SV-C2_847296.csv": "India",
    "fire_archive_SV-C2_847301.csv": "Nigeria",
    "fire_archive_SV-C2_847305.csv": "DR Congo",
}
ORIGINAL = ["South Asia", "Southeast Asia", "North and Central America"]
COUNTRY = {"South Asia": "Pakistan", "Southeast Asia": "Indonesia", "North and Central America": "United States",
           "Brazil": "Brazil", "Australia": "Australia", "India": "India", "Nigeria": "Nigeria",
           "DR Congo": "Democratic Republic of the Congo"}
