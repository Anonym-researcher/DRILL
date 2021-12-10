from ontolearn import KnowledgeBase, LearningProblemGenerator, DrillAverage
from ontolearn import Experiments
from ontolearn.binders import DLLearnerBinder
import pandas as pd
from argparse import ArgumentParser
import os
import json
import time
import gradio as gr
import torch
import random
full_computation_time = time.time()


def deploy(args):
    """

    """
    print('Parse input knowledge base')
    kb = KnowledgeBase(args.path_knowledge_base)
    print('Load pretrained model')
    drill_average = DrillAverage(pretrained_model_path=args.pretrained_drill_avg_path, knowledge_base=kb,
                                 path_of_embeddings=args.path_knowledge_base_embeddings,
                                 verbose=args.verbose, num_workers=args.num_workers)
    print('Load URIs of individuals for random learning problems')
    uris_individuals= [ i  for i in drill_average.instance_embeddings.index.tolist() if 'http://www.benchmark.org/family#' in i]

    def fit(positive_examples, negative_examples, size_of_examples, random_examples: bool):
        if random_examples:
            # Either sample from here self.instance_idx_mapping
            # or sample from targets
            pos_str = random.sample(uris_individuals, int(size_of_examples))
            neg_str = random.sample(uris_individuals, int(size_of_examples))
        else:
            pos_str = positive_examples.split(",")
            neg_str = negative_examples.split(",")
        dataset = [("unknown", {_ for _ in pos_str}, {_ for _ in neg_str})]
        with torch.no_grad():
            report = drill_average.fit_from_iterable(dataset=dataset, max_runtime=1)
            # report List of  {'Target': target_concept_str,'Prediction': h.concept.name,'F-measure': f_measure,'Accuracy': accuracy,'NumClassTested': self.quality_func.num_times_applied,'Runtime': rn}

        if len(pos_str) < 20:
            s = f'E^+:{",".join(pos_str)}\nE^-:{",".join(neg_str)}\n'
        else:
            s = f'|E^+|:{len(pos_str)}\n|E^-|:{len(neg_str)}\n'
        return s, pd.DataFrame(report)

    gr.Interface(
        fn=fit,
        inputs=[gr.inputs.Textbox(lines=5, placeholder=None, label='Positive Examples'),
                gr.inputs.Textbox(lines=5, placeholder=None, label='Negative Examples'),
                gr.inputs.Slider(minimum=1, maximum=100),
                "checkbox"],
        outputs=[gr.outputs.Textbox(label='Learning Problem'), gr.outputs.Dataframe(label='Predictions')],
        title='Deploy DRILL for Rapid Class Expression Learning',
        description='Click Random Examples & Submit.').launch()


if __name__ == '__main__':
    # General
    parser = ArgumentParser()
    # LP dependent
    parser.add_argument("--path_knowledge_base", type=str, default='KGs/Family/family-benchmark_rich_background.owl')
    parser.add_argument("--path_knowledge_base_embeddings", type=str,
                        default='embeddings/ConEx_Family/ConEx_entity_embeddings.csv')
    parser.add_argument('--pretrained_drill_avg_path', type=str,
                        help='pre_trained_agents/Family/DrillHeuristic_averaging/DrillHeuristic_averaging.pth')
    # Concept Learning Testing
    parser.add_argument("--iter_bound", type=int, default=10_000, help='iter_bound during testing.')
    parser.add_argument('--max_test_time_per_concept', type=int, default=3, help='Max. runtime during testing')

    # General
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=32, help='Number of cpus used during batching')
    deploy(parser.parse_args())
