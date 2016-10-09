#include "logreg.h"
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <stdint.h>

using namespace std;

int INDEX_SIZE = 1 << 18;

void skipEmptyAndComment(ifstream& file, string& s) {
	do {
		getline(file, s);
	} while (s.size() == 0 || s[0] == '%');
}

void splitStringAndHash(string s, const char delimiter, vector<size_t>& x, double& y) {
    const char* c = s.c_str();
    uint64_t hash = 17;
    uint64_t hash2 = 17;
    bool first = true;
    bool group = false;
    
    y = (atoi(s.substr(0, 2).c_str()) == 1) ? 1 : -1;

    while(*c) {
        while(*c && *c == delimiter) { ++c; }
        if(!*c) break;
        if(*c == '|') {
            group = true;
            hash = 17;
        }
        hash2 = hash;
        while(*c && *c != delimiter) { 
            hash2 = hash2 * 31 + *c;
            ++c;
        }
        if(group) { 
            hash = hash2; 
            group = false; 
        } else {
            if(first) {
                first = false;
                continue;
            }
            x.push_back(hash2 & (INDEX_SIZE - 1));
        }
    }
} 

LogisticRegressionProblem::LogisticRegressionProblem(const char* filename) {
    numFeats = INDEX_SIZE;
	ifstream matfile(filename);
	if (!matfile.good()) {
		cerr << "error opening matrix file " << filename << endl;
		exit(1);
	}
    printf("step1\n");
    string str;
    vector<size_t> x;
    double y;
    while(!matfile.eof()) {
        getline(matfile, str);
        if(str.size() < 3) break;
        x.clear();
        splitStringAndHash(str, ' ', x, y);
	    AddInstance(x, y > 0 ? true:false);
    }
    printf("instance end end %d \n", instance_starts[labels.size() - 1]);
    printf("finish\n");
}

LogisticRegressionProblem::LogisticRegressionProblem(const char* matFilename, const char* labelFilename) {
	ifstream matfile(matFilename);
	if (!matfile.good()) {
		cerr << "error opening matrix file " << matFilename << endl;
		exit(1);
	}
	string s;
	getline(matfile, s);
	if (!s.compare("%%MatrixMarket matrix coordinate real general")) {
		skipEmptyAndComment(matfile, s);

		stringstream st(s);
		size_t numIns, numNonZero;
		st >> numIns >> numFeats >> numNonZero;

		vector<deque<size_t> > rowInds(numIns);
		vector<deque<float> > rowVals(numIns);
		for (size_t i = 0; i < numNonZero; i++) {
			size_t row, col;
			float val;
			matfile >> row >> col >> val;
			row--;
			col--;
			rowInds[row].push_back(col);
			rowVals[row].push_back(val);
		}

		matfile.close();

		ifstream labfile(labelFilename);
		getline(labfile, s);
		if (s.compare("%%MatrixMarket matrix array real general")) {
			cerr << "unsupported label file format in " << labelFilename << endl;
			exit(1);
		}

		skipEmptyAndComment(labfile, s);
		stringstream labst(s);
		size_t labNum, labCol;
		labst >> labNum >> labCol;
		if (labNum != numIns) {
			cerr << "number of labels doesn't match number of instances in " << labelFilename << endl;
			exit(1);
		} else if (labCol != 1) {
			cerr << "label matrix may not have more than one column" << endl;
			exit(1);
		}

		instance_starts.push_back(0);

		for (size_t i=0; i<numIns; i++) {
			int label;
			labfile >> label;
			bool bLabel;
			switch (label) {
					case 1:
						bLabel = true;
						break;

					case -1:
						bLabel = false;
						break;

					default:
						cerr << "illegal label: must be 1 or -1" << endl;
						exit(1);
			}

			AddInstance(rowInds[i], rowVals[i], bLabel);
		}

		labfile.close();
	} else if (!s.compare("%%MatrixMarket matrix array real general")) {
		skipEmptyAndComment(matfile, s);
		stringstream st(s);
		size_t numIns;
		st >> numIns >> numFeats;

		vector<vector<float> > rowVals(numIns);

		for (size_t j=0; j<numFeats; j++) {
			for (size_t i=0; i<numIns; i++) {
				float val;
				matfile >> val;
				rowVals[i].push_back(val);
			}
		}

		matfile.close();

		ifstream labfile(labelFilename);
		getline(labfile, s);
		if (s.compare("%%MatrixMarket matrix array real general")) {
			cerr << "unsupported label file format in " << labelFilename << endl;
			exit(1);
		}

		skipEmptyAndComment(labfile, s);
		stringstream labst(s);
		size_t labNum, labCol;
		labst >> labNum >> labCol;
		if (labNum != numIns) {
			cerr << "number of labels doesn't match number of instances in " << labelFilename << endl;
			exit(1);
		} else if (labCol != 1) {
			cerr << "label matrix may not have more than one column" << endl;
			exit(1);
		}

		instance_starts.push_back(0);
		for (size_t i=0; i<numIns; i++) {
			int label;
			labfile >> label;
			bool bLabel;
			switch (label) {
					case 1:
						bLabel = true;
						break;

					case -1:
						bLabel = false;
						break;

					default:
						cerr << "illegal label: must be 1 or -1" << endl;
						exit(1);
			}

			AddInstance(rowVals[i], bLabel);
		}

		labfile.close();
	} else {
		cerr << "unsupported matrix file format in " << matFilename << endl;
		exit(1);
	}
}

void LogisticRegressionProblem::AddInstance(const deque<size_t>& inds, const deque<float>& vals, bool label) {
	for (size_t i=0; i<inds.size(); i++) {
		indices.push_back(inds[i]);
		values.push_back(vals[i]);
	}
	instance_starts.push_back(indices.size());
	labels.push_back(label);
}

void LogisticRegressionProblem::AddInstance(const vector<size_t>& inds, bool label) {
	for (size_t i=0; i<inds.size(); i++) {
		indices.push_back(inds[i]);
		values.push_back(1);
		//values.push_back(vals[i]);
	}
	instance_starts.push_back(indices.size());
	labels.push_back(label);
}

void LogisticRegressionProblem::AddInstance(const vector<float>& vals, bool label) {
	for (size_t i=0; i<vals.size(); i++) {
		values.push_back(vals[i]);
	}
	instance_starts.push_back(values.size());
	labels.push_back(label);
}

double LogisticRegressionProblem::ScoreOf(size_t i, const vector<double>& weights) const {
	double score = 0;
	for (size_t j=instance_starts[i]; j < instance_starts[i+1]; j++) {
		double value = values[j];
		//size_t index = (indices.size() > 0) ? indices[j] : j - instance_starts[i];
		size_t index = indices[j];
		score += weights[index] * value;
	}
	if (!labels[i]) score *= -1;
	return score;
}


double LogisticRegressionObjective::Eval(const DblVec& input, DblVec& gradient) {
	double loss = 1.0;

    int start,end;
    start = clock();
	for (size_t i=0; i<input.size(); i++) {
		loss += 0.5 * input[i] * input[i] * l2weight;
		gradient[i] = l2weight * input[i];
	}
    end = clock();
    //printf("eval step1 used: %f\n", (end - start) / (double) CLOCKS_PER_SEC);

    start = clock();
    vector<double> scores;
	for (size_t i =0 ; i<problem.NumInstances(); i++) {
        scores.push_back(-problem.ScoreOf(i, input));
    }
	for (size_t i =0 ; i<problem.NumInstances(); i++) {
		double insLoss, insProb;
	    double temp = 1.0 + exp(scores[i]);
		insLoss = log(temp);
		insProb = 1.0/temp;
		loss += insLoss;
		problem.AddMultTo(i, 1.0 - insProb, gradient);
    }
	//for (size_t i =0 ; i<problem.NumInstances(); i++) {
		//double score = problem.ScoreOf(i, input);

		//double insLoss, insProb;
	    //double temp = 1.0 + exp(-score);
		//insLoss = log(temp);
		//insProb = 1.0/temp;
		//if (score < -30) {
		//	insLoss = -score;
		//	insProb = 0;
		//} else if (score > 30) {
		//	insLoss = 0;
		//	insProb = 1;
		//} else {
		//	double temp = 1.0 + exp(-score);
		//	insLoss = log(temp);
		//	insProb = 1.0/temp;
		//}
		//loss += insLoss;
		//problem.AddMultTo(i, 1.0 - insProb, gradient);
	//}
    end = clock();
    //printf("eval step2 used: %f\n", (end - start) / (double) CLOCKS_PER_SEC);

	return loss;
}
