from covid_q import train, test
from features import build_features, server as dependency_server
from nltk.classify.maxent import train_maxent_classifier_with_iis
from nltk.parse import corenlp
import atexit

dependency_server.start()
atexit.register(dependency_server.stop)

with open('log', 'w') as log:
    
        # Build features data for MaxEnt classifier
        training_data = train.apply(build_features, axis=1)


        # Train classifier
        ment = train_maxent_classifier_with_iis(training_data)

        correct = 0
        incorrect = 0

        i = 0

        # Build features for test data
        for features, label in test.apply(build_features, axis=1):
            found_label = ment.classify(features)
            log.write(f'{test.loc[i, "Question"]}\n')
            if found_label == label: 
                correct += 1
                log.write(f'\t{feat_string} -> {found_label}\n')
            else:
                incorrect += 1
                feat_string = ''.join([f'{key}={val} ' for key,val in features.items()])
                log.write(f'!\t{feat_string} -!> {found_label}, {label}\n')
            log.write('\n')
            i += 1

        print(f'Final accuracy: {correct/incorrect}')
