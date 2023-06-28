from sentence_transformers import SentenceTransformer, LoggingHandler
import logging
import pandas as pd

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

import numpy as np
 
english = pd.read_pickle("./dataset_statistics/statistics_english_sent.pkl", )
doc = np.array_split(english, 5) #.iloc[:30]

if __name__ == '__main__':
    n = 0
    for df in doc:
        #Create a large list of 100k sentences
        df = df.reset_index(drop = True)
        sentences = df.sentence.tolist()
        print(len(sentences))

        #Define the model
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

        #Start the multi-process pool on all available CUDA devices
        pool = model.start_multi_process_pool()

        #Compute the embeddings using the multi-process pool
        emb = model.encode_multi_process(sentences, pool)
        print("Embeddings computed. Shape:", emb.shape)
        
        final = pd.concat([df, pd.DataFrame(emb, index=list(range(len(df))))], axis=1)
        final.to_csv("./dataset_statistics/english_sent_embedding_" + str(n) + ".csv")
        #Optional: Stop the proccesses in the pool
        n = n +1
        print("fold: ", n)
    model.stop_multi_process_pool(pool)