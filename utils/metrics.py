from sklearn.metrics import accuracy_score, f1_score, classification_report


def get_metrics(outputs, targets):
    accuracy = accuracy_score(targets, outputs)
    micro_f1 = f1_score(targets, outputs, average='micro')
    macro_f1 = f1_score(targets, outputs, average='macro')
    return accuracy, micro_f1, macro_f1

def get_classification_report(outputs, targets, labels):
    report = classification_report(targets, outputs, target_names=labels)
    return report