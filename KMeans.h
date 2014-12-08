/*
 * KMeans.h
 *
 *  Created on: Nov 23, 2014
 *      Author: Tom
 */

#ifndef KMEANS_H_
#define KMEANS_H_

#include <vector>

namespace ML {

class KMeans {

public:

	KMeans();

	~KMeans();

	struct featureVec {
		std::vector<float> data;
		featureVec(int len) { data.resize(len, 0.f); }
		featureVec() { featureVec(0); }
		void resize(unsigned int s) { data.resize(s); }
		unsigned int size() const { return data.size(); }
		float operator[](int i) const { return data[i]; }
		float &operator[](int i) { return data[i]; }
		featureVec operator+(const featureVec &other) const;
		void operator+=(const featureVec &other);
		featureVec operator/(float f) const;
		void operator/=(float f);
		void operator=(float f);
		featureVec &operator=(const featureVec &rhs);
		featureVec operator*(const featureVec &other) const;
	};

	typedef std::vector<featureVec> dataSet;

	void train(int k);

	unsigned int classify(const featureVec &vec) const;

	unsigned int classify(const std::vector<dataSet> &data, const dataSet &features) const;

	const dataSet &getCentroids() const { return centroids; }

	void setTrainingData(const dataSet &data);

	void setDistortionFn(float (*distortion)(const featureVec&, const featureVec&)) { distortionFn = distortion; }

private:

	float minDistortion(const dataSet &centroids, const featureVec &vector) const;
	float minDistortion(const dataSet &centroids, const featureVec &vector, int *classification) const;
	float averageMinDistortion(const dataSet &centroids, const dataSet &vectors) const;
	void updateCentroids();
	int getNearestCluster(const dataSet &centroids, const featureVec &feature) const;
	void partitionFeatures(const dataSet &centroids, const dataSet &features);

	int k;
	dataSet centroids;
	dataSet trainingData;
	std::vector<int> classification;
	float deltaThresh;
	float (*distortionFn)(const featureVec&, const featureVec&);

	static float euclidSQ(const featureVec &v1, const featureVec &v2);
	static float sum(const std::vector<float> &vec);
};

}

#endif /* KMEANS_H_ */
