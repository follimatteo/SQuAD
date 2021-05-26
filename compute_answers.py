from bidaf import bidaf_model, DistAccuracy, Similarity, Q2CAttention, C2QAttention, MergedContext
import json
import pandas as pd
import tensorflow as tf

import argparse
'''
parser = argparse.ArgumentParser(description='.')
parser.add_argument('--path', metavar='N', type=str, nargs='+',
                    help='path to json test set')

args = parser.parse_args()
print(args)
'''

### funzione per leggere il json
print("loading dataset")
dataframerows = []

with open("./datasets/test_set.json") as f:
    data = json.load(f)

    for el in data['data']:
        title = el['title']
        paragraphs = el['paragraphs']

        for context_qas in paragraphs:
              context = context_qas['context']
              qas = context_qas['qas']

              for a_q in qas: #[0]["answers"][0]["text"]
                  answer = a_q['answers']
                  question = a_q['question']
                  id = a_q['id']

                  # CREATE ROW
                  row = {
                      "id" : id,
                      "context" : context,
                      "context_list": context.split(" "),
                      "question" : question,
                      "question_list": question.split(" ")
                      }

                  # APPEND ROW TO DATAFRAME ROWS
                  dataframerows.append(row)

df_test = pd.DataFrame(dataframerows)
df_test = df_test[["id", "context", "context_list", "question", "question_list"]]


print("loading model")
keras_model = tf.keras.models.load_model(
    "h5/Bidaf.D-ATT.h5", custom_objects={"DistAccuracy": DistAccuracy, "Similarity": Similarity, "C2QAttention":C2QAttention, "Q2CAttention": Q2CAttention, "MergedContext" : MergedContext}, compile=True)

print("loading bidaf utils")
BiDAF = bidaf_model(keras_model, "./vocabularies/VocabCHAR.json.")

print("making prediction")
bidaf_prediction = BiDAF.predict(df_test)

print(predicted_answers)
