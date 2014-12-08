/*
 * KMeans.cpp
 *
 *  Created on: Nov 23, 2014
 *      Author: Tom
 */

#include "KMeans.h"

#include <vector>
#include <set>
#include <cfloat>
#include <cstdlib>

using namespace ML;

KMeans::KMeans() {
	deltaThresh = 0.01f;
	k = 0;
	distortionFn = euclidSQ;
}

KMeans::~KMeans() {}

void KMeans::train(int k) {

	this->k = k;
	centroids.resize(k);

	// Select random initial vectors
	std::set<int> indices;
	for (int i = 0; i < k; ++i) {
		int num;
		do {
			num = rand() % trainingData.size();
		} while (indices.find(num) != indices.end());
		indices.insert(num);
		centroids[i] = trainingData[num];
	}

	// Calculate initial distortion
	float oldDist = averageMinDistortion(centroids, trainingData);

	// Iterate until solution converges
	while (1) {

		// Partition data by closest centroid
		partitionFeatures(centroids, trainingData);

		// Generate new centroids
		updateCentroids();

		// Calculate new distortion
		float newDist = averageMinDistortion(centroids, trainingData);

		// Check for progress stall
		if ((oldDist - newDist)/oldDist < deltaThresh)
			break;
		oldDist = newDist;

	}

}

unsigned int KMeans::classify(const featureVec &vec) const {

	float min = FLT_MAX, dist;
	unsigned int c = 0;
	for (int i = 0; i < k; ++i) {
		dist = distortionFn(centroids[i], vec);
		if (dist < min) {
			min = dist;
			c = i;
		}
	}

	return c;

}

unsigned int KMeans::classify(const std::vector<dataSet> &codebooks, const dataSet &features) const {

	float min = FLT_MAX, dist;
	unsigned int c = 0;
	for (unsigned int i = 0; i < codebooks.size(); ++i) {
		const dataSet &codebook = codebooks[i];
		dist = 0.f;
		for (unsigned int j = 0; j < features.size(); ++j) {
			dist += minDistortion(codebook, features[j]);
		}
		if (dist < min) {
			min = dist;
			c = i;
		}
	}

	return c;

}

void KMeans::updateCentroids() {

	int count;
	for (int cluster = 0; cluster < k; ++cluster) {
		centroids[cluster] = 0.f;
		count = 0;
		for (unsigned int i = 0; i < classification.size(); ++i) {
			if (classification[i] == cluster) {
				centroids[cluster] += trainingData[i];
				++count;
			}
		}
		centroids[cluster] /= float(count);
	}

}

int KMeans::getNearestCluster(const dataSet &centroids, const featureVec &feature) const {

	int c;
	minDistortion(centroids, feature, &c);
	return c;

}

void KMeans::setTrainingData(const dataSet &data) {

	trainingData = data;
	classification.resize(data.size(), -1);

}

float KMeans::minDistortion(const dataSet &centroids, const featureVec &feature) const {

	float min = FLT_MAX, dist;
	for (int i = 0; i < k; ++i) {
		dist = distortionFn(centroids[i], feature);
		if (dist < min)
			min = dist;
	}

	return min;
}

float KMeans::minDistortion(const dataSet &centroids, const featureVec &feature, int *classification) const {

	float min = FLT_MAX, dist;
	for (int i = 0; i < k; ++i) {
		dist = distortionFn(centroids[i], feature);
		if (dist < min) {
			min = dist;
			*classification = i;
		}
	}

	return min;
}

float KMeans::averageMinDistortion(const dataSet &centroids, const dataSet &vectors) const {

	float sum = 0.f;

	for (unsigned int i = 0; i < vectors.size(); ++i)
		sum += minDistortion(centroids, vectors[i]);

	return sum/vectors.size();

}

void KMeans::partitionFeatures(const dataSet &centroids, const dataSet &features) {

	for (unsigned int i = 0; i < classification.size(); ++i)
		classification[i] = getNearestCluster(centroids, features[i]);

}

KMeans::featureVec KMeans::featureVec::operator+(const featureVec &other) const {

	featureVec vec(data.size());
	for (unsigned int i = 0; i < data.size(); ++i)
		vec[i] = data[i] + other.data[i];
	return vec;

}

void KMeans::featureVec::operator+=(const featureVec &other) {

	for (unsigned int i = 0; i < data.size(); ++i)
		data[i] += other.data[i];

}

KMeans::featureVec KMeans::featureVec::operator/(float f) const {

	featureVec tmp(data.size());
	for (unsigned int i = 0; i < data.size(); ++i)
		tmp.data[i] = data[i] / f;
	return tmp;

}

void KMeans::featureVec::operator/=(float f) {

	for (std::vector<float>::iterator it = data.begin(); it != data.end(); ++it)
		*it /= f;

}

void KMeans::featureVec::operator=(float f) {

	for (std::vector<float>::iterator it = data.begin(); it != data.end(); ++it)
		*it = f;

}

KMeans::featureVec &KMeans::featureVec::operator=(const featureVec &rhs) {

	data = rhs.data;
	return *this;

}

KMeans::featureVec KMeans::featureVec::operator*(const featureVec &other) const {

	featureVec vec(data.size());
	for (unsigned int i = 0; i < data.size(); ++i)
		vec[i] = data[i] * other.data[i];
	return vec;

}

float KMeans::euclidSQ(const featureVec &v1, const featureVec &v2) {

	float sum = 0.f;
	for (unsigned int i = 0; i < v1.size(); ++i)
		sum += (v1[i]-v2[i])*(v1[i]-v2[i]);
	return sum;

}

float KMeans::sum(const std::vector<float> &vec) {

	float sum = 0.f;
	for (std::vector<float>::const_iterator it = vec.begin(); it != vec.end(); ++it)
		sum += *it;
	return sum;

}
