
# Select Datasets and Embeddings
family_dataset_path=$PWD'/KGs/Family/family-benchmark_rich_background.owl'
# Embeddings
family_kge=$PWD'/embeddings/ConEx_Family/ConEx_entity_embeddings.csv'

num_episode=100 #denoted by M in the manuscript
min_num_concepts=3 #denoted by n in the manuscript
num_of_randomly_created_problems_per_concept=2 # denoted by m in the manuscript
echo "Training Starts"
python drill_train.py --path_knowledge_base "$family_dataset_path" --min_length 3 --num_of_sequential_actions 4 --path_knowledge_base_embeddings "$family_kge" --num_episode $num_episode --min_num_concepts $min_num_concepts --num_of_randomly_created_problems_per_concept $num_of_randomly_created_problems_per_concept
echo "Training Ends"