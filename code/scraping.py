import requests
import pandas as pd


url = "https://remoteok.com/api"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)
data = response.json()

jobs = data[1:]

df = pd.DataFrame(jobs)
df = df[["position", "tags", "description"]]
df = df.dropna(subset=["tags"])
df["skills"] = df["tags"].apply(lambda x: ", ".join([tag.lower() for tag in x]))
df = df.rename(columns={"position": "job_title"})
df_final = df[["job_title", "skills", "description"]]

print(df_final.head())
print("Total jobs :", len(df_final))

df_final.to_csv("data/csv/remoteok_jobs_columns.csv", index=False)