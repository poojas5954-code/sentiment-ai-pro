import pandas as pd, os, random
os.makedirs("data", exist_ok=True)

positives = ["i am happy","i am very happy","i am great","i am awesome","i am amazing","feeling happy","feeling great","this is wonderful","this is amazing","i love this","i am excited","life is beautiful","best day ever","so happy","fantastic","excellent","superb","i am joyful","i am delighted","i am cheerful","love it","wonderful day","amazing day","feeling blessed","so good","very good","great job","perfect","outstanding"]

negatives = ["i am sad","i am very sad","i am depressed","i am angry","i am upset","i am miserable","i am lonely","i am stressed","feeling sad","feeling bad","this is bad","this is terrible","i hate this","worst day","horrible","awful","so bad","very bad","disappointed","hate it","life is hard","today is bad","i am anxious","i am irritated","i am gloomy","feeling low","not happy","depressed","angry"]

neutrals = ["i am fine","i am okay","i am alright","it is okay","it is fine","average day","so so","nothing special","normal day","regular day","i am neutral","it is average","okay okay","fine","neutral","as usual","usual day","nothing new","it is okayish","just okay","middle","moderate","not bad not good","balanced","calm","stable","ordinary","common day","just fine","i am good"]

data = []
for _ in range(18):
    for t in positives: data.append([t,"positive"])
    for t in negatives: data.append([t,"negative"])
    for t in neutrals: data.append([t,"neutral"])

random.shuffle(data)
df = pd.DataFrame(data, columns=["text","label"])
df.to_csv("data/dataset.csv", index=False)
print(f"Created {len(df)} rows: {df['label'].value_counts().to_dict()}")