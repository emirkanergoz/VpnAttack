import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

df = pd.read_parquet("new_train_data.parquet")
print(df.head())

# ATTACK TIME

df['tarih'] = pd.to_datetime(df['attack_time']).dt.date
df['saat'] = pd.to_datetime(df['attack_time']).dt.hour

def zaman_dilimi(saat):
    if 6 <= saat < 12:
        return "Sabah"
    elif 12 <= saat < 18:
        return "Öğlen"
    elif 18 <= saat < 24:
        return "Akşam"
    else:
        return "Gece"

df['zaman_dilimi'] = df['saat'].apply(zaman_dilimi)

# Zaman dilimlerine göre label dağılımı
zaman_dilimi_distribution = df.groupby(['zaman_dilimi', 'label']).size().unstack(fill_value=0)

# Vpn olan atakların tüm ataklara oranı /
zaman_dilimi_distribution['label_1_ratio'] = zaman_dilimi_distribution[1] / (
    zaman_dilimi_distribution[0] + zaman_dilimi_distribution[1]
)

label_1_ratio_map = {
    "Akşam": 0.056,
    "Gece": 0.053,
    "Sabah": 0.058,
    "Öğlen": 0.053,
}
df.drop("attack_time", axis=1, inplace=True)

df['attack_time'] = df['zaman_dilimi'].map(label_1_ratio_map)

df.drop(columns = ["tarih","saat","zaman_dilimi"],axis=1,inplace=True)




# WATCHER_COUNTRY

wc_mode = df["watcher_country"].mode()[0]
df["watcher_country"] = df["watcher_country"].fillna(wc_mode)
# print(df["watcher_country"].nunique()) # 113 farklı ülke va

# watcher_country ve attacker_country için Target Encoding
mean_encoded_watcher = df.groupby('watcher_country', observed=False)['label'].mean()
df['watcher_country_encoded'] = df['watcher_country'].map(mean_encoded_watcher)

mean_encoded_attacker = df.groupby('attacker_country', observed=False)['label'].mean()
df['attacker_country_encoded'] = df['attacker_country'].map(mean_encoded_attacker)

df.drop(columns = ["watcher_country", "attacker_country"], axis=1 , inplace=True)

#as_name
df.drop(columns=["watcher_as_name","attacker_as_name"],axis = 1 , inplace=True)


#attack_type one hot coding
attack_type = pd.get_dummies(df["attack_type"],dtype=int)
df = pd.concat([df,attack_type],axis=1)

label = df["label"]

df.drop(columns=["label","attack_type"],axis = 1, inplace=True)

df = pd.concat([df,label],axis=1)


#
aan_mean = df["attacker_as_num"].mean()
df["attacker_as_num"] = df["attacker_as_num"].fillna(aan_mean)



ac_mode = df["attacker_country_encoded"].mode()[0]
df["attacker_country_encoded"] = df["attacker_country_encoded"].fillna(ac_mode)

X = df.drop(columns=["label"])  # Özellikler (features)
y = df["label"]  # Etiketler (target)

print(df.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
#tahmin yapma
y_pred = knn.predict(X_test_scaled)
#başarı hesaplama
accuracy = accuracy_score(y_test, y_pred)
# Sonuçları yazdırma
print(f"Başarı Oranı (Accuracy): {accuracy * 100:.2f}%")



