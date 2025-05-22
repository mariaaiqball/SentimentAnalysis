import pandas as pd 

'''CHANGE CSV'''
sb_df = pd.read_csv('sb.csv')
sb_df['label'] = ''
tweet_counter = 0 

for i, row in sb_df.iterrows():
    print(f"\nText {i+1}/{len(sb_df)}: {row['text']}")
    label = input("Label (positive/negative/neutral): ").strip().lower()

    while label not in ['positive', 'negative', 'neutral']:
        print("Invalid label. Please enter 'positive', 'negative', or 'neutral'.")
        label = input("Label (positive/negative/neutral): ").strip().lower()

    sb_df.at[i, 'label'] = label
    tweet_counter+=1

sb_df.to_csv('labeled_output.csv', index=False)
print(f"Completed labeling {tweet_counter} rows")
