/*
 * driver.cpp
 *
 *  Created on: Nov 25, 2014
 *      Author: Tom
 */

#include <cstdio>
#include <iostream>
#include <vector>
#include <cstdlib>

#include "KMeans.h"
using namespace std;

typedef std::vector<ML::KMeans::dataSet> spdata;

spdata readSpeakerData(FILE *f) {

	int len;
	fread(&len, sizeof(len), 1, f);

	int r, c;

	spdata sp;
	sp.resize(len);
	for (int i = 0; i < len; ++i) {
		fread(&r, sizeof(int), 1, f);
		fread(&c, sizeof(int), 1, f);
		sp[i].resize(c);
		for (int j = 0; j < c; ++j) {
			sp[i][j].resize(r);
			for (int k = 0; k < r; ++k) {
				fread(&sp[i][j][k], sizeof(float), 1, f);
			}
		}
	}

	return sp;

}

int main(int argc, char *argv[]) {

	int k;
	if (argc == 1)
		k = 10;
	else if (argc == 2)
		k = atoi(argv[1]);
	else {
		cerr << "Usage: " << argv[0] << " <k>" << endl;
		return 1;
	}

	cout << "Reading data..." << flush;

	FILE *f = fopen("D:\\Documents\\Work\\Fall 14\\CS 682\\f_sp_train.dat", "rb");
	if (!f) {
		cout << "Failed to open file" << endl;
		return 1;
	}
	spdata train = readSpeakerData(f);
	fclose(f);

	f = fopen("D:\\Documents\\Work\\Fall 14\\CS 682\\f_sp_test.dat", "rb");
	if (!f) {
		cout << "Failed to open file" << endl;
		return 1;
	}
	spdata test = readSpeakerData(f);
	fclose(f);

	cout << " done." << endl;

	// Train
	cout << "Training..." << flush;
	ML::KMeans kmeans;
	spdata codebooks;
	codebooks.reserve(train.size());
	for (unsigned int i = 0; i < train.size(); ++i) {
		kmeans.setTrainingData(train[i]);
		kmeans.train(k);
		codebooks.push_back(kmeans.getCentroids());
	}
	cout << " done." << endl;

	// Test
	cout << "Testing..." << flush;
	int correct = 0, incorrect = 0;
	for (unsigned int i = 0; i < test.size(); ++i) {
		unsigned int c = kmeans.classify(codebooks, test[i]);
		if (c != i)
			incorrect++;
		else
			correct++;
	}
	cout << " done." << endl;

	// Summary
	cout << "Total tests: " << correct+incorrect << endl;
	cout << "Correct: " << correct << endl;
	cout << "Incorrect: " << incorrect << endl;
	cout << "Error rate: " << (float)incorrect/(correct+incorrect) << endl;

	return 0;
}


