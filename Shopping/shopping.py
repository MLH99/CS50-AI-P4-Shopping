import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer 0
        - Administrative_Duration, a floating point number 1
        - Informational, an integer 2
        - Informational_Duration, a floating point number 3
        - ProductRelated, an integer 4
        - ProductRelated_Duration, a floating point number 5
        - BounceRates, a floating point number 6
        - ExitRates, a floating point number 7
        - PageValues, a floating point number 8
        - SpecialDay, a floating point number 9
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer 11
        - Browser, an integer 12
        - Region, an integer 13
        - TrafficType, an integer 14
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    
    file = open("shopping.csv", "r")
    
    list_evidence = list()
    list_labels = list()
    
    with open("shopping.csv", "r") as f:
        header = f.readline()
        
        for line in f:
            
            list_cells = list()
            label = None
            
            columns = line.split(",")
            for i in range(17):
                cell = columns[i]
                
                append = True
                
                if i in (0,2,4,11,12,13,14):
                    cell = int(cell)
                elif i in (1,3,5,6,7,8,9):
                    cell = float(cell)
                elif i == 15:
                    if cell == "Returning_Visitor":
                        cell = 1
                    else:
                        cell = 0
                elif i == 10:
                    if cell == "Jan":
                        cell = 0
                    elif cell == "Feb":
                        cell = 1
                    elif cell == "Mar":
                        cell = 2
                    elif cell == "Apr":
                        cell = 3
                    elif cell == "May":
                        cell = 4
                    elif cell == "June":
                        cell = 5
                    elif cell == "Jul":
                        cell = 6
                    elif cell == "Aug":
                        cell = 7
                    elif cell == "Sep":
                        cell = 8
                    elif cell == "Oct":
                        cell = 9
                    elif cell == "Nov":
                        cell = 10
                    elif cell == "Dec":
                        cell = 11
                        
                elif i == 16:
                    if cell == "TRUE":
                        cell = 1
                    else:
                        cell = 0
                    label = cell
                    append = False
                
                if append:
                    list_cells.append(cell)
            
            list_evidence.append(list_cells)
            list_labels.append(label)
                    

    list_data = (list_evidence, list_labels)
    
    return list_data

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    
    knn = KNeighborsClassifier(n_neighbors=1)
    
    knn.fit(evidence, labels)
    
    return knn


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    
    correct_true_negative = 0
    correct_true_positive = 0
    count_true = 0
    count_false = 0
    
    for evidence, prediction in zip(labels, predictions):
        if evidence == 1:
            count_true += 1
            if prediction == 1:
                correct_true_positive +=1
        elif evidence == 0:
            count_false += 1
            if prediction == 0:
                correct_true_negative += 1
    
    sensitivity = 0
    specificity = 0
    
    if count_false > 0 and count_true > 0:
        sensitivity = correct_true_positive / count_true
        specificity = correct_true_negative / count_false 
    
    return_tuple = (sensitivity, specificity)
    
    return return_tuple


if __name__ == "__main__":
    main()
