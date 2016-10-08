#include <iostream>
#include <deque>
#include <fstream>

#include "OWLQN.h"
#include "leastSquares.h"
#include "logreg.h"

using namespace std;

void printUsageAndExit() {
	cout << "Orthant-Wise Limited-memory Quasi-Newton trainer" << endl;
	cout << "trains L1-regularized logistic regression or least-squares models" << endl << endl;
	cout << "usage: feature_file label_file regWeight output_file [options]" << endl;
	cout << "  feature_file   input feature matrix in Matrix Market format (mxn real coordinate or array)" << endl;
	cout << "                   rows represent features for each instance" << endl;
	cout << "  label_file     input instance labels in Matrix Market format (mx1 real array)" << endl;
	cout << "                   rows contain single real value" << endl;
	cout << "                   for logistic regression problems, value must be 1 or -1" << endl;
	cout << "  regWeight      coefficient of l1 regularizer" << endl;
	cout << "  output_file    output weight vector in Matrix Market format (1xm real array)" << endl << endl;
	cout << "options:" << endl;
	cout << "  -ls            use least squares formulation (logistic regression is default)" << endl;
	cout << "  -q             quiet.  Suppress all output" << endl;
	cout << "  -tol <value>   sets convergence tolerance (default is 1e-4)" << endl;
	cout << "  -m <value>     sets L-BFGS memory parameter (default is 10)" << endl;
	cout << "  -l2weight <value>" << endl;
	cout << "                 sets L2 regularization weight (default is 0)" << endl;
	cout << endl;
	exit(0);
}

void printVector(const DblVec &vec, const char* filename) {
	ofstream outfile(filename);
	if (!outfile.good()) {
		cerr << "error opening matrix file " << filename << endl;
		exit(1);
	}
	outfile << "%%MatrixMarket matrix array real general" << endl;
	outfile << "1 " << vec.size() << endl;
	for (size_t i=0; i<vec.size(); i++) {
		outfile << vec[i] << endl;
	}
	outfile.close();
}

int main(int argc, char* argv[]) {

	if (argc < 5 || !strcmp(argv[1], "-help") || !strcmp(argv[1], "--help") ||
		!strcmp(argv[1], "-h") || !strcmp(argv[1], "-usage")) {
			printUsageAndExit();
	}

	const char* feature_file = argv[1];
	const char* label_file = argv[2];
	double regweight = atof(argv[3]);
	const char* output_file = argv[4];

	if (regweight < 0) {
		cout << "L1 regularization weight must be non-negative." << endl;
		exit(1);
	}

	bool leastSquares = false, quiet = false;
	double tol = 1e-4, l2weight = 0;
	int m = 10;

	for (int i=5; i<argc; i++) {
		if (!strcmp(argv[i], "-ls")) leastSquares = true;
		else if (!strcmp(argv[i], "-q")) quiet = true;
		else if (!strcmp(argv[i], "-tol")) {
			++i;
			if (i >= argc || (tol = atof(argv[i])) <= 0) {
				cout << "-tol (convergence tolerance) flag requires 1 positive real argument." << endl;
				exit(1);
			}
		} else if (!strcmp(argv[i], "-l2weight")) {
			++i;
			if (i >= argc || (l2weight = atof(argv[i])) < 0) {
				cout << "-l2weight flag requires 1 non-negative real argument." << endl;
				exit(1);
			}
		}	else if (!strcmp(argv[i], "-m")) {
			++i;
			if (i >= argc || (m = atoi(argv[i])) == 0) {
				cout << "-m (L-BFGS memory param) flag requires 1 positive int argument." << endl;
				exit(1);
			}
		} else {
			cerr << "unrecognized argument: " << argv[i] << endl;
			exit(1);
		}
	}

	if (!quiet) {
		cout << argv[0] << " called with arguments " << endl << "   ";
		for (int i=1; i<argc; i++) {
			cout << argv[i] << " ";
		}
		cout << endl;
	}

	DifferentiableFunction *obj;
	size_t size;
	if (leastSquares) {
		LeastSquaresProblem *prob = new LeastSquaresProblem(feature_file, label_file);
		obj = new LeastSquaresObjective(*prob, l2weight);
		size = prob->NumFeats(); 
	} else {
		LogisticRegressionProblem *prob = new LogisticRegressionProblem(feature_file, label_file);
		obj = new LogisticRegressionObjective(*prob, l2weight);
		size = prob->NumFeats(); 
	}

	DblVec init(size), ans(size);

	OWLQN opt(quiet);
	opt.Minimize(*obj, init, ans, regweight, tol, m);

	int nonZero = 0;
	for (size_t i = 0; i<ans.size(); i++) {
		if (ans[i] != 0) nonZero++;
	}

	if (!quiet) cout << "Finished with optimization.  " << nonZero << "/" << size << " non-zero weights." << endl;

	printVector(ans, output_file);

	return 0;
}
