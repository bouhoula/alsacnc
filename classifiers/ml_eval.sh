if [ -e "data/AA_purpose_detection.csv" ]; then
    echo "Evaluating declared purpose detection model"
    python classifiers/text_classifiers/kfold.py --model_type Bert --model bert-base-uncased --tokenizer bert-base-uncased --task purpose_detection --data data/AA_purpose_detection.csv --num_epochs 10 --learning_rate 4e-5
else
    echo "Declared purpose detection model: dataset missing. Skipping evaluation."
fi

if [ -e "data/interactive_elements.csv" ]; then
    echo "Evaluating interactive elements model"
    python classifiers/text_classifiers/kfold.py --model_type Bert --model bert-base-uncased --tokenizer bert-base-uncased --task ie_text_classification --data data/interactive_elements.csv --use_loss_weights
else
    echo "Interactive elements model: dataset missing. Skipping evaluation."
fi

echo "Evaluating cookie classification. Preprocessing..."

python classifiers/cookieblock_classifier/train_eval_per_website.py --crawled_data data/consentomatic-reject-tranco05May-20210514_081933.json --output_dir data/reject_all_data --prepare_data

python classifiers/cookieblock_classifier/train_eval_per_website.py --crawled_data data/tranco_05May_20210510_201615.json --output_dir data/accept_all_data --prepare_data

python classifiers/cookieblock_classifier/train_eval_per_website.py --crawled_data data/consentomatic-reject-tranco05May-20210514_081933.json --output_dir data/reject_all_data_cookiebot --prepare_data --ignore_cmp 1 --ignore_cmp 2

python classifiers/cookieblock_classifier/train_eval_per_website.py --crawled_data data/tranco_05May_20210510_201615.json --output_dir data/accept_all_data_onetrust_termly --prepare_data --ignore_cmp 0


echo '\nTraining on all CMPs, evaluating on Cookiebot'
python classifiers/cookieblock_classifier/train_eval_per_website.py --accept_all_dir data/accept_all_data --reject_all_dir data/reject_all_data


echo '\nTraining on Onetrust + Termly, evaluating on Cookiebot'
python classifiers/cookieblock_classifier/train_eval_per_website.py --accept_all_dir data/accept_all_data_onetrust_termly --reject_all_dir data/reject_all_data_cookiebot



