from covid_q import train, test
from features import build_features, server as dependency_server
from nltk.classify.maxent import train_maxent_classifier_with_iis
from nltk.parse import corenlp
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC
import atexit

scikit = SklearnClassifier(LinearSVC())
dependency_server.start()
atexit.register(dependency_server.stop)

with open('log', 'w') as log:
    
        # Build features data for MaxEnt classifier
        training_data = train.apply(build_features, axis=1)


        # Train classifier
        scikit.train(training_data)
        ment = train_maxent_classifier_with_iis(training_data)

        correct = 0
        correct_svm = 0
        incorrect = 0
        incorrect_svm = 0

        # Build features for test data
        for idx, (features, label) in enumerate(test.apply(build_features, axis=1)):
            found_label_svm ,= scikit.classify_many([features])
            found_label = ment.classify(features)

            log.write(f'{test.loc[idx, "Question"]}\n')
            feat_string = ''.join([f'{key}={val} ' for key,val in features.items()])
            if found_label == label: 
                correct += 1
                log.write(f'\t{feat_string} -> {found_label}\n')
            else:
                incorrect += 1
                log.write(f'!\t{feat_string} -!> {found_label}, {label}\n')
            
            if found_label_svm == label:
                correct_svm += 1
            else:
                incorrect_svm += 1

            log.write('\n')
        print(f'Final accuracy (maxent): {correct/incorrect}')
        print(f'Final accuracy (svm): {correct_svm/incorrect_svm}')
