'''
Date : January 2017

Authors : Zhiming Luo and Pierre-Marc Jodoin from the University of Sherbrooke

Description : code used to compute classification metrics of the MIO-TCD
            classification dataset.  Please see http://tcd.miovision.com/
            for more details

Execution : simply type the following command in a terminal:

   >> python classification_evaluation.py gt_train.csv your_results_train.csv

NOTE: this code was developed and tested with Python 3.5.2 and Linux (Ubuntu 14.04)

Disclamer:

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''

from __future__ import print_function
import csv
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import sys

classes = ['articulated_truck', 'bus', 'car', 'work_van', 'background']


def compute_metric(result, gt, classes):
    '''
    Computes the following metrics with scikit-learn :
           * confusion matrix
           * cohen kappa score
           * precision score
           * recall score
           * accuracy

     Both 'result' and 'gt' are a python dictionary for which 'image name' is
     the key and 'label' is the value, ex:
           gt['00008854.jpg'] = 'bicycle'
     '''
    y_true = []
    y_pred = []

    try:
        for img1, img2 in zip(gt.keys(),result.keys()):
            y_true.append(gt[img1])
            y_pred.append(result[img2])
    except Exception as e:
        print(e)

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cks = cohen_kappa_score(y_true, y_pred, labels=classes)
    ps = precision_score(y_true, y_pred, average=None)
    rs = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    metrics = {}
    metrics['confusion matrix'] = cm
    metrics['cohen kappa score'] = cks
    metrics['precision score'] = ps
    metrics['recall score'] = rs
    metrics['f1 score'] = f1
    metrics['mean recall'] = np.mean(rs)
    metrics['mean precision'] = np.mean(ps)
    metrics['mean f1'] = np.mean(f1)
    metrics['accuracy'] = np.diagonal(cm).sum() / float(len(y_true))
    return metrics


def csv_evalution(gt_file, res_file, classes):
    '''
    Computes the classification scores between gt_file and res_file.  These
    are csv files each containing at each line, a file name with its associated
    label.  gt_file typically contains groundtruth and res_file results from
    a given algorithm. For example:

    (...)
    00260164,non-motorized_vehicle
    00375878,car
    00692348,background
    (...)

    Please see "gt_train.csv" for an example.
    '''

    gt = {}
    with open(gt_file, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            gt[row[0]] = row[1]

    results = {}
    with open(res_file, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        results = dict(reader)

    metrics = compute_metric(results, gt, classes)

    return metrics

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("Usage : \n\t python classification_evaluation.py gt_train.csv your_results_train.csv")
    else:
        print('Computing score between ', sys.argv[1], ' and ', sys.argv[2], '\n')
        metrics = csv_evalution(sys.argv[1], sys.argv[2], classes)

        for key in metrics.keys():
            print(key + ':')
            print(metrics[key])
            print('\n')
