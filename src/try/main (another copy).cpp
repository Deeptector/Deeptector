#include <iostream>
using namespace std;
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
#include "run_darknet.h"
#include <tuple>
#include <dirent.h>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fstream>
#ifdef _DEBUG
  #undef _DEBUG
  #include <python3.5m/Python.h>
  #define _DEBUG
#else
  #include <python3.5m/Python.h>
#endif
#include <python3.5m/numpy/arrayobject.h>
#define POSE_MAX_PEOPLE 96
#define NET_OUT_CHANNELS 57 // 38 for pafs, 19 for part


template<typename T>
inline int intRound(const T a)
{
	return int(a+0.5f);
}

template<typename T>
inline T fastMin(const T a, const T b)
{
	return (a < b ? a : b);
}

void render_pose_keypoints
	(
	Mat& frame,
	const vector<float>& keypoints,
	vector<int> keyshape,
	const float threshold,
	float scale
	)
{
	const int num_keypoints = keyshape[1];
	unsigned int pairs[] =
	{
		1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10,
		1, 11, 11, 12, 12, 13, 1, 0, 0, 14, 14, 16, 0, 15, 15, 17
	};
	float colors[] =
	{
		255.f, 0.f, 85.f, 255.f, 0.f, 0.f, 255.f, 85.f, 0.f, 255.f, 170.f, 0.f,
		255.f, 255.f, 0.f, 170.f, 255.f, 0.f, 85.f, 255.f, 0.f, 0.f, 255.f, 0.f,
		0.f, 255.f, 85.f, 0.f, 255.f, 170.f, 0.f, 255.f, 255.f, 0.f, 170.f, 255.f,
		0.f, 85.f, 255.f, 0.f, 0.f, 255.f, 255.f, 0.f, 170.f, 170.f, 0.f, 255.f,
		255.f, 0.f, 255.f, 85.f, 0.f, 255.f
	};
	const int pairs_size = sizeof(pairs) / sizeof(unsigned int);
	const int number_colors = sizeof(colors) / sizeof(float);

	for (int person = 0; person < keyshape[0]; ++person)
	{
		// Draw lines
		for (int pair = 0u; pair < pairs_size; pair += 2)
		{
			const int index1 = (person * num_keypoints + pairs[pair]) * keyshape[2];
			const int index2 = (person * num_keypoints + pairs[pair + 1]) * keyshape[2];
			if (keypoints[index1 + 2] > threshold && keypoints[index2 + 2] > threshold)
			{
				const int color_index = pairs[pair + 1] * 3;
				Scalar color { colors[(color_index + 2) % number_colors],
					colors[(color_index + 1) % number_colors],
					colors[(color_index + 0) % number_colors]};
				Point keypoint1{ intRound(keypoints[index1] * scale), intRound(keypoints[index1 + 1] * scale) };
				Point keypoint2{ intRound(keypoints[index2] * scale), intRound(keypoints[index2 + 1] * scale) };
				line(frame, keypoint1, keypoint2, color, 2);
			}
		}
		// Draw circles
		for (int part = 0; part < num_keypoints; ++part)
		{
			const int index = (person * num_keypoints + part) * keyshape[2];
			if (keypoints[index + 2] > threshold)
			{
				const int color_index = part * 3;
				Scalar color { colors[(color_index + 2) % number_colors],
					colors[(color_index + 1) % number_colors],
					colors[(color_index + 0) % number_colors]};
				Point center{ intRound(keypoints[index] * scale), intRound(keypoints[index + 1] * scale) };
				circle(frame, center, 3, color, -1);
			}
		}
	}
}

void connect_bodyparts
	(
	vector<float>& pose_keypoints,
	const float* const map,
	const float* const peaks,
	int mapw,
	int maph,
	const int inter_min_above_th,
	const float inter_th,
	const int min_subset_cnt,
	const float min_subset_score,
	vector<int>& keypoint_shape
	)
{
	keypoint_shape.resize(3);
	const int body_part_pairs[] =
	{
		1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 1, 11, 11,
		12, 12, 13, 1, 0, 0, 14, 14, 16, 0, 15, 15, 17, 2, 16, 5, 17
	};
	const int limb_idx[] =
	{
		31, 32, 39, 40, 33, 34, 35, 36, 41, 42, 43, 44, 19, 20, 21, 22, 23, 24, 25,
		26, 27, 28, 29, 30, 47, 48, 49, 50, 53, 54, 51, 52, 55, 56, 37, 38, 45, 46
	};
	const int num_body_parts = 18; // COCO part number
	const int num_body_part_pairs = num_body_parts + 1;
	std::vector<std::pair<std::vector<int>, double>> subset;
	const int subset_counter_index = num_body_parts;
	const int subset_size = num_body_parts + 1;
	const int peaks_offset = 3 * (POSE_MAX_PEOPLE + 1);
	const int map_offset = mapw * maph;

	for (unsigned int pair_index = 0u; pair_index < num_body_part_pairs; ++pair_index)
	{
		const int body_partA = body_part_pairs[2 * pair_index];
		const int body_partB = body_part_pairs[2 * pair_index + 1];
		const float* candidateA = peaks + body_partA*peaks_offset;
		const float* candidateB = peaks + body_partB*peaks_offset;
		const int nA = (int)(candidateA[0]); // number of part A candidates
		const int nB = (int)(candidateB[0]); // number of part B candidates

		// add parts into the subset in special case
		if (nA == 0 || nB == 0)
		{
			// Change w.r.t. other
			if (nA == 0) // nB == 0 or not
			{
				for (int i = 1; i <= nB; ++i)
				{
					bool num = false;
					for (unsigned int j = 0u; j < subset.size(); ++j)
					{
						const int off = body_partB*peaks_offset + i * 3 + 2;
						if (subset[j].first[body_partB] == off)
						{
							num = true;
							break;
						}
					}
					if (!num)
					{
						std::vector<int> row_vector(subset_size, 0);
						// store the index
						row_vector[body_partB] = body_partB*peaks_offset + i * 3 + 2;
						// the parts number of that person
						row_vector[subset_counter_index] = 1;
						// total score
						const float subsetScore = candidateB[i * 3 + 2];
						subset.emplace_back(std::make_pair(row_vector, subsetScore));
					}
				}
			}
			else // if (nA != 0 && nB == 0)
			{
				for (int i = 1; i <= nA; i++)
				{
					bool num = false;
					for (unsigned int j = 0u; j < subset.size(); ++j)
					{
						const int off = body_partA*peaks_offset + i * 3 + 2;
						if (subset[j].first[body_partA] == off)
						{
							num = true;
							break;
						}
					}
					if (!num)
					{
						std::vector<int> row_vector(subset_size, 0);
						// store the index
						row_vector[body_partA] = body_partA*peaks_offset + i * 3 + 2;
						// parts number of that person
						row_vector[subset_counter_index] = 1;
						// total score
						const float subsetScore = candidateA[i * 3 + 2];
						subset.emplace_back(std::make_pair(row_vector, subsetScore));
					}
				}
			}
		}
		else // if (nA != 0 && nB != 0)
		{
			std::vector<std::tuple<double, int, int>> temp;
			const int num_inter = 10;
			// limb PAF x-direction heatmap
			const float* const mapX = map + limb_idx[2 * pair_index] * map_offset;
			// limb PAF y-direction heatmap
			const float* const mapY = map + limb_idx[2 * pair_index + 1] * map_offset;
			// start greedy algorithm
			for (int i = 1; i <= nA; i++)
			{
				for (int j = 1; j <= nB; j++)
				{
					const int dX = candidateB[j * 3] - candidateA[i * 3];
					const int dY = candidateB[j * 3 + 1] - candidateA[i * 3 + 1];
					const float norm_vec = float(std::sqrt(dX*dX + dY*dY));
					// If the peaksPtr are coincident. Don't connect them.
					if (norm_vec > 1e-6)
					{
						const float sX = candidateA[i * 3];
						const float sY = candidateA[i * 3 + 1];
						const float vecX = dX / norm_vec;
						const float vecY = dY / norm_vec;
						float sum = 0.;
						int count = 0;
						for (int lm = 0; lm < num_inter; lm++)
						{
							const int mX = fastMin(mapw - 1, intRound(sX + lm*dX / num_inter));
							const int mY = fastMin(maph - 1, intRound(sY + lm*dY / num_inter));
							const int idx = mY * mapw + mX;
							const float score = (vecX*mapX[idx] + vecY*mapY[idx]);
							if (score > inter_th)
							{
								sum += score;
								++count;
							}
						}

						// parts score + connection score
						if (count > inter_min_above_th)
						{
							temp.emplace_back(std::make_tuple(sum / count, i, j));
						}
					}
				}
			}
			// select the top minAB connection, assuming that each part occur only once
			// sort rows in descending order based on parts + connection score
			if (!temp.empty())
			{
				std::sort(temp.begin(), temp.end(), std::greater<std::tuple<float, int, int>>());
			}
			std::vector<std::tuple<int, int, double>> connectionK;

			const int minAB = fastMin(nA, nB);
			// assuming that each part occur only once, filter out same part1 to different part2
			std::vector<int> occurA(nA, 0);
			std::vector<int> occurB(nB, 0);
			int counter = 0;
			for (unsigned int row = 0u; row < temp.size(); row++)
			{
				const float score = std::get<0>(temp[row]);
				const int aidx = std::get<1>(temp[row]);
				const int bidx = std::get<2>(temp[row]);
				if (!occurA[aidx - 1] && !occurB[bidx - 1])
				{
					// save two part score "position" and limb mean PAF score
					connectionK.emplace_back(std::make_tuple(body_partA*peaks_offset + aidx * 3 + 2,
						body_partB*peaks_offset + bidx * 3 + 2, score));
					++counter;
					if (counter == minAB)
					{
						break;
					}
					occurA[aidx - 1] = 1;
					occurB[bidx - 1] = 1;
				}
			}
			// Cluster all the body part candidates into subset based on the part connection
			// initialize first body part connection
			if (pair_index == 0)
			{
				for (const auto connectionKI : connectionK)
				{
					std::vector<int> row_vector(num_body_parts + 3, 0);
					const int indexA = std::get<0>(connectionKI);
					const int indexB = std::get<1>(connectionKI);
					const double score = std::get<2>(connectionKI);
					row_vector[body_part_pairs[0]] = indexA;
					row_vector[body_part_pairs[1]] = indexB;
					row_vector[subset_counter_index] = 2;
					// add the score of parts and the connection
					const double subset_score = peaks[indexA] + peaks[indexB] + score;
					subset.emplace_back(std::make_pair(row_vector, subset_score));
				}
			}
			// Add ears connections (in case person is looking to opposite direction to camera)
			else if (pair_index == 17 || pair_index == 18)
			{
				for (const auto& connectionKI : connectionK)
				{
					const int indexA = std::get<0>(connectionKI);
					const int indexB = std::get<1>(connectionKI);
					for (auto& subsetJ : subset)
					{
						auto& subsetJ_first = subsetJ.first[body_partA];
						auto& subsetJ_first_plus1 = subsetJ.first[body_partB];
						if (subsetJ_first == indexA && subsetJ_first_plus1 == 0)
						{
							subsetJ_first_plus1 = indexB;
						}
						else if (subsetJ_first_plus1 == indexB && subsetJ_first == 0)
						{
							subsetJ_first = indexA;
						}
					}
				}
			}
			else
			{
				if (!connectionK.empty())
				{
					for (unsigned int i = 0u; i < connectionK.size(); ++i)
					{
						const int indexA = std::get<0>(connectionK[i]);
						const int indexB = std::get<1>(connectionK[i]);
						const double score = std::get<2>(connectionK[i]);
						int num = 0;
						// if A is already in the subset, add B
						for (unsigned int j = 0u; j < subset.size(); j++)
						{
							if (subset[j].first[body_partA] == indexA)
							{
								subset[j].first[body_partB] = indexB;
								++num;
								subset[j].first[subset_counter_index] = subset[j].first[subset_counter_index] + 1;
								subset[j].second = subset[j].second + peaks[indexB] + score;
							}
						}
						// if A is not found in the subset, create new one and add both
						if (num == 0)
						{
							std::vector<int> row_vector(subset_size, 0);
							row_vector[body_partA] = indexA;
							row_vector[body_partB] = indexB;
							row_vector[subset_counter_index] = 2;
							const float subsetScore = peaks[indexA] + peaks[indexB] + score;
							subset.emplace_back(std::make_pair(row_vector, subsetScore));
						}
					}
				}
			}
		}
	}

	// Delete people below thresholds, and save to output
	int number_people = 0;
	std::vector<int> valid_subset_indexes;
	valid_subset_indexes.reserve(fastMin((size_t)POSE_MAX_PEOPLE, subset.size()));
	for (unsigned int index = 0; index < subset.size(); ++index)
	{
		const int subset_counter = subset[index].first[subset_counter_index];
		const double subset_score = subset[index].second;
		if (subset_counter >= min_subset_cnt && (subset_score / subset_counter) > min_subset_score)
		{
			++number_people;
			valid_subset_indexes.emplace_back(index);
			if (number_people == POSE_MAX_PEOPLE)
			{
				break;
			}
		}
	}

	// Fill and return pose_keypoints
	keypoint_shape = { number_people, (int)num_body_parts, 3 };
	if (number_people > 0)
	{
		pose_keypoints.resize(number_people * (int)num_body_parts * 3);
	}
	else
	{
		pose_keypoints.clear();
	}
	for (unsigned int person = 0u; person < valid_subset_indexes.size(); ++person)
	{
		const auto& subsetI = subset[valid_subset_indexes[person]].first;
		for (int bodyPart = 0u; bodyPart < num_body_parts; bodyPart++)
		{
			const int base_offset = (person*num_body_parts + bodyPart) * 3;
			const int body_part_index = subsetI[bodyPart];
			if (body_part_index > 0)
			{
				pose_keypoints[base_offset] = peaks[body_part_index - 2];
				pose_keypoints[base_offset + 1] = peaks[body_part_index - 1];
				pose_keypoints[base_offset + 2] = peaks[body_part_index];
			}
			else
			{
				pose_keypoints[base_offset] = 0.f;
				pose_keypoints[base_offset + 1] = 0.f;
				pose_keypoints[base_offset + 2] = 0.f;
			}
		}
	}
}

void find_heatmap_peaks
	(
	const float *src,
	float *dst,
	const int SRCW,
	const int SRCH,
	const int SRC_CH,
	const float TH
	)
{
	// find peaks (8-connected neighbor), weights with 7 by 7 area to get sub-pixel location and response
	const int SRC_PLANE_OFFSET = SRCW * SRCH;
	// add 1 for saving total people count, 3 for x, y, score
	const int DST_PLANE_OFFSET = (POSE_MAX_PEOPLE + 1) * 3;
	float *dstptr = dst;
	int c = 0;
	int x = 0;
	int y = 0;
	int i = 0;
	int j = 0;
	// TODO: reduce multiplication by using pointer
	for(c = 0; c < SRC_CH - 1; ++c)
	{
		int num_people = 0;
		for(y = 1; y < SRCH - 1 && num_people != POSE_MAX_PEOPLE; ++y)
		{
			for(x = 1; x < SRCW - 1 && num_people != POSE_MAX_PEOPLE; ++x)
			{
				int idx  = y * SRCW + x;
				float value = src[idx];
				if (value > TH)
				{
					const float TOPLEFT = src[idx - SRCW - 1];
					const float TOP = src[idx - SRCW];
					const float TOPRIGHT = src[idx - SRCW + 1];
					const float LEFT = src[idx - 1];
					const float RIGHT = src[idx + 1];
					const float BUTTOMLEFT = src[idx + SRCW - 1];
					const float BUTTOM = src[idx + SRCW];
					const float BUTTOMRIGHT = src[idx + SRCW + 1];
					if(value > TOPLEFT && value > TOP && value > TOPRIGHT && value > LEFT &&
						value > RIGHT && value > BUTTOMLEFT && value > BUTTOM && value > BUTTOMRIGHT)
					{
						float x_acc = 0;
						float y_acc = 0;
						float score_acc = 0;
						for (i = -3; i <= 3; ++i)
						{
							int ux = x + i;
							if (ux >= 0 && ux < SRCW)
							{
								for (j = -3; j <= 3; ++j)
								{
									int uy = y + j;
									if (uy >= 0 && uy < SRCH)
									{
										float score = src[uy * SRCW + ux];
										x_acc += ux * score;
										y_acc += uy * score;
										score_acc += score;
									}
								}
							}
						}
						x_acc /= score_acc;
						y_acc /= score_acc;
						score_acc = value;
						dstptr[(num_people + 1) * 3 + 0] = x_acc;
						dstptr[(num_people + 1) * 3 + 1] = y_acc;
						dstptr[(num_people + 1) * 3 + 2] = score_acc;
						++num_people;
					}
				}
			}
		}
		dstptr[0] = num_people;
		src += SRC_PLANE_OFFSET;
		dstptr += DST_PLANE_OFFSET;
	}
}

Mat create_netsize_im
	(
	const Mat &im,
	const int netw,
	const int neth,
	float *scale
	)
{
	// for tall image
	int newh = neth;
	float s = newh / (float)im.rows;
	int neww = im.cols * s;
	if (neww > netw)
	{
		//for fat image
		neww = netw;
		s = neww / (float)im.cols;
		newh = im.rows * s;
	}

	*scale = 1 / s;
	Rect dst_area(0, 0, neww, newh);
	Mat dst = Mat::zeros(neth, netw, CV_8UC3);
	resize(im, dst(dst_area), Size(neww, newh));
	return dst;
}

bool standardization(std::vector<float> *v)
{

    float max_value = *(std::max_element(v->begin(), v->end()));
    float min_value = *(std::min_element(v->begin(), v->end()));

 

    float range = 1.0/(max_value - min_value);

    std::vector<float>::iterator pos = v->begin();

    std::vector<float>::iterator end = v->end();
    int idx = 0;

    for(; pos != end; pos++, idx++){
	if(idx % 3 != 2)
		*pos = (*pos - min_value)*range;

    } 
    return true; 
}

void pythonEmbedding()
{
// python embedding
    cout << "embedding start\n";
    PyObject *pModule;
    PyObject *pModuleName;
    PyObject *pModuleFunc;
    char pyFileName[100];
    strcpy(pyFileName, "rnn");

    // See [5]
    const wchar_t c_s[] = L":/home/macoy/projects/pythonTesting/main";
    wchar_t *s = new wchar_t[sizeof(c_s) / sizeof(c_s[0])];
    wcscpy(s, c_s);
    const wchar_t p_s[] = L":.";
    wchar_t *ps = new wchar_t[sizeof(p_s) / sizeof(p_s[0])];
    wcscpy(ps, p_s);

    //Py_SetProgramName(L"test"); // see [4]

    Py_Initialize();

    wchar_t **changed_argv;
    changed_argv = (wchar_t **)malloc(sizeof*changed_argv);
    
    changed_argv[0] = (wchar_t *)malloc(strlen(pyFileName) + 1);
    mbstowcs(changed_argv[0], pyFileName, strlen(pyFileName) + 1);
        
    PySys_SetArgv(0, (wchar_t**)changed_argv);
    PyRun_SimpleString("import tensorflow as tf\n");


    delete[] s;

    // See [6]; The below code makes it so we can import from the current directory
    wchar_t *path, *newpath;
    path=Py_GetPath();
    newpath=new wchar_t[wcslen(path)+4];
    wcscpy(newpath, path);
    wcscat(newpath, ps);  // ":." for unix, or ";." for windows
    PySys_SetPath(newpath);

    delete[] newpath;


    // See [3]
    pModuleName = PyUnicode_FromString(pyFileName);
    pModule = PyImport_Import(pModuleName);
    Py_DECREF(pModuleName);

/*
    PyObject* pArgs = NULL;
    PyObject *pReturnVal = NULL;

    // Follow [1] to get what's going on
    if (pModule != NULL)
    {
    pModuleFunc = PyObject_GetAttrString(pModule, "init");
        if (pModuleFunc && PyCallable_Check(pModuleFunc))
        {
        pArgs = NULL;
        pReturnVal = PyObject_CallObject(pModuleFunc, pArgs);
        //Py_DECREF(pArgs);
	//Py_DECREF(pReturnVal);
        }
        else{
            if(PyErr_Occurred())
        	PyErr_Print();
        std::cout << "error: no func\n";
        }   
	Py_XDECREF(pModuleFunc);
        Py_DECREF(pModule);
    	}
    else
    {
    PyErr_Print();
    std::cout << "\nerror: no module\n";
    }
    //free(pArgs);
    Py_CLEAR(pReturnVal);
    free(changed_argv[0]);
    free(changed_argv);
    //Py_Finalize();
*/

	cout << "embedding end\n";

}

PyObject* vectorToList_Float(const vector<vector<float>> &data) {
	PyObject* listObj = PyList_New(16);
	if (!listObj) throw logic_error("Unable to allocate memory for Python list");
	/*for(int i=0; i<16; i++) {
		for(int j=0; j<54; j++) {
			cout << data.at(i).at(j) << ", ";
		}
		cout << endl;
	}*/
	for (int i=0; i <16; i++) {
		PyObject* obj = PyList_New(54);
		for(int j=0; j<54; j++) {
			PyObject *num = PyFloat_FromDouble( (double) data.at(i).at(j));
			if (!num) {
				Py_DECREF(listObj);
				throw logic_error("Unable to allocate memory for Python list");
			}
			PyList_SET_ITEM(obj, j, num);
		}
		PyList_SET_ITEM(listObj, i, obj);
	}
	return listObj;
}

int main
	(
	int ac,
	char **av
	)
{
	if (ac < 5)
	{
		cout << "usage: ./bin [input_type] [image file] [cfg file] [weight file]" << endl;
		cout << "input_type -> [image] [video] [rtsp]" << endl;
		return 1;
	}

	// 1. read args
	char *input_type = av[1];
	char *im_path = av[2];
	char *cfg_path = av[3];
	char *weight_path = av[4];
	char *train_type = av[5];

	//pythonEmbedding();
	PyObject *pModule;
	PyObject *pModuleName;
	PyObject *pModuleFunc1;
	PyObject *pModuleFunc2;
	PyObject *pModuleFunc3;
	char pyFileName[100];
	strcpy(pyFileName, "rnn");

	Py_Initialize();
	import_array()

	wchar_t **changed_argv;
	changed_argv = (wchar_t **)malloc(sizeof*changed_argv);
	changed_argv[0] = (wchar_t *)malloc(strlen(pyFileName) + 1);
	mbstowcs(changed_argv[0], pyFileName, strlen(pyFileName) + 1);
        
	PySys_SetArgv(0, (wchar_t**)changed_argv);
	PyRun_SimpleString("import tensorflow as tf\n");

	pModuleName = PyUnicode_FromString(pyFileName);
	pModule = PyImport_Import(pModuleName);
	Py_DECREF(pModuleName);

	PyObject* pArgs1 = NULL;
	PyObject* pArgs2 = NULL;
	PyObject* pArgs3 = NULL;
        PyObject *pReturnVal1 = NULL;
	PyObject *pReturnVal2 = NULL;
	PyObject *pReturnVal3 = NULL;

        if (pModule != NULL)
        { 
	pModuleFunc3 = PyObject_GetAttrString(pModule, "python_init");
	    if (pModuleFunc3 && PyCallable_Check(pModuleFunc3))
            {
            pArgs3 = NULL;
            pReturnVal3 = PyObject_CallObject(pModuleFunc3, pArgs3);
            }
            else{
                if(PyErr_Occurred())
        	    PyErr_Print();
            std::cout << "error: no func\n";
            }
	    Py_XDECREF(pModuleFunc3);
            //Py_DECREF(pModule);
    	    }
        else
        {
        PyErr_Print();
        std::cout << "\nerror: no module\n";
        }

	if(strcmp(input_type, "image") == 0) {
		Mat im = imread(im_path);
		namedWindow("result", 1);  
		if (im.empty())
		{
			cout << "failed to read image" << endl;
			return 1;
		}

		// 2. initialize net
		int net_inw = 0;
		int net_inh = 0;
		int net_outw = 0;
		int net_outh = 0;
		init_net(cfg_path, weight_path, &net_inw, &net_inh, &net_outw, &net_outh);

		// 3. resize to net input size, put scaled image on the top left
		float scale = 0.0f;
		Mat netim = create_netsize_im(im, net_inw, net_inh, &scale);

		// 4. normalized to float type
		netim.convertTo(netim, CV_32F, 1 / 256.f, -0.5);

		// 5. split channels
		float *netin_data = new float[net_inw * net_inh * 3]();
		float *netin_data_ptr = netin_data;
		vector<Mat> input_channels;
		for (int i = 0; i < 3; ++i)
		{
			Mat channel(net_inh, net_inw, CV_32FC1, netin_data_ptr);
			input_channels.emplace_back(channel);
			netin_data_ptr += (net_inw * net_inh);
		}
		split(netim, input_channels);

		// 6. feed forward
		double time_begin = getTickCount();
		float *netoutdata = run_net(netin_data);
		double fee_time = (getTickCount() - time_begin) / getTickFrequency() * 1000;
		cout << "forward fee: " << fee_time << "ms" << endl;

		// 7. resize net output back to input size to get heatmap
		float *heatmap = new float[net_inw * net_inh * NET_OUT_CHANNELS];
		for (int i = 0; i < NET_OUT_CHANNELS; ++i)
		{
			Mat netout(net_outh, net_outw, CV_32F, (netoutdata + net_outh*net_outw*i));
			Mat nmsin(net_inh, net_inw, CV_32F, heatmap + net_inh*net_inw*i);
			resize(netout, nmsin, Size(net_inw, net_inh), 0, 0, CV_INTER_CUBIC);
		}

		// 8. get heatmap peaks
		float *heatmap_peaks = new float[3 * (POSE_MAX_PEOPLE+1) * (NET_OUT_CHANNELS-1)];
		find_heatmap_peaks(heatmap, heatmap_peaks, net_inw, net_inh, NET_OUT_CHANNELS, 0.05);

		// 9. link parts
		vector<float> keypoints;
		vector<int> shape;
		connect_bodyparts(keypoints, heatmap, heatmap_peaks, net_inw, net_inh, 9, 0.05, 6, 0.4, shape);

		// 10. draw result
		render_pose_keypoints(im, keypoints, shape, 0.05, scale);

		// prediction!!!!!
		if (pModule != NULL)
        	{ 
		pModuleFunc1 = PyObject_GetAttrString(pModule, "action_classification");
	    	if (pModuleFunc1 && PyCallable_Check(pModuleFunc1))
            	{
            	pArgs1 = NULL;
            	pReturnVal1 = PyObject_CallObject(pModuleFunc1, pArgs1);
            	}
            	else{
             	   if(PyErr_Occurred())
        		    PyErr_Print();
            	std::cout << "error: no func\n";
            	}
	    	Py_XDECREF(pModuleFunc3);
            	//Py_DECREF(pModule);
    	 	   }
       	 	else
        	{
        	PyErr_Print();
        	std::cout << "\nerror: no module\n";
       		}

		// 11. show and save result
		cout << "people: " << shape[0] << endl;
		imshow("result", im); //show image
		imwrite("output/result.jpg",im);//save result as jpg
		waitKey(0);
		delete [] heatmap_peaks;
		delete [] heatmap;
		delete [] netin_data;


	}

	else if(strcmp(input_type, "video") == 0) {

		Mat im;
		namedWindow("result", 1); 
		VideoCapture cap(im_path);  
		vector<vector<float>> vecs;
		vector<float> vec;
		vector<vector<float>> inputs;
		if (!cap.isOpened())  
		{  
			cout << "failed to read video" << endl;
		} 
		Size size = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),(int)cap.get(CAP_PROP_FRAME_HEIGHT));
		VideoWriter outputVideo;
		outputVideo.open("ouput.avi", VideoWriter::fourcc('X', 'V', 'I', 'D'),
			30, size, true);
		cap.read(im);
		imshow("result", im);
		if (im.empty())
		{
			cout << "failed to read image (" << im_path << ")" << endl;
		}

		// 2. initialize net
		int net_inw = 0;
		int net_inh = 0;
		int net_outw = 0;
		int net_outh = 0;
		init_net(cfg_path, weight_path, &net_inw, &net_inh, &net_outw, &net_outh);

		while(!im.empty()) {

			// 3. resize to net input size, put scaled image on the top left
			float scale = 0.0f;
			Mat netim = create_netsize_im(im, net_inw, net_inh, &scale);

			// 4. normalized to float type
			netim.convertTo(netim, CV_32F, 1 / 256.f, -0.5);

			// 5. split channels
			float *netin_data = new float[net_inw * net_inh * 3]();
			float *netin_data_ptr = netin_data;
			vector<Mat> input_channels;
			for (int i = 0; i < 3; ++i)
			{
				Mat channel(net_inh, net_inw, CV_32FC1, netin_data_ptr);
				input_channels.emplace_back(channel);
				netin_data_ptr += (net_inw * net_inh);
			}
			split(netim, input_channels);

			// 6. feed forward
			double time_begin = getTickCount();
			float *netoutdata = run_net(netin_data);
			double fee_time = (getTickCount() - time_begin) / getTickFrequency() * 1000;
			cout << "forward fee: " << fee_time << "ms" << endl;

			// 7. resize net output back to input size to get heatmap
			float *heatmap = new float[net_inw * net_inh * NET_OUT_CHANNELS];
			for (int i = 0; i < NET_OUT_CHANNELS; ++i)
			{
				Mat netout(net_outh, net_outw, CV_32F, (netoutdata + net_outh*net_outw*i));
				Mat nmsin(net_inh, net_inw, CV_32F, heatmap + net_inh*net_inw*i);
				resize(netout, nmsin, Size(net_inw, net_inh), 0, 0, CV_INTER_CUBIC);
			}

			// 8. get heatmap peaks
			float *heatmap_peaks = new float[3 * (POSE_MAX_PEOPLE+1) * (NET_OUT_CHANNELS-1)];
			find_heatmap_peaks(heatmap, heatmap_peaks, net_inw, net_inh, NET_OUT_CHANNELS, 0.05);

			// 9. link parts
			vector<float> keypoints;
			vector<int> shape;
			connect_bodyparts(keypoints, heatmap, heatmap_peaks, net_inw, net_inh, 9, 0.05, 6, 0.4, shape);
			//printf("KeyPoint Count = %d\n", keypoints.size());
			/*for(int i=0; i<keypoints.size(); i++) {
				printf("%f ", keypoints[i]);
			}
			printf("\n");*/
			// 10. draw result
			render_pose_keypoints(im, keypoints, shape, 0.05, scale);
			if(!im.empty()) {
				imshow("result", im); //show image
				waitKey(1);
			}
			// 사람이 1명 있을 때 
			if(shape[0] = 1) {
				string str_inputs = "";
				inputs.clear();
				vecs.push_back(keypoints);
				for(int k=0; k<vecs.size(); k++) 
					standardization(&vecs.at(k));
				if(vecs.size() == 17) {
					for(int i=0; i<vecs.size() - 1; i++) {
						vector<float> input;
						for(int j=0; j<54; j++) {
							
							if(j % 3 == 2) {
								str_inputs = str_inputs + to_string((vecs.at(i).at(j) + vecs.at(i+1).at(j)) / 2.0) + ",";
								input.push_back((vecs.at(i).at(j) + vecs.at(i+1).at(j)) / 2.0);
							} 
							else {
								if(vecs.at(i).at(j) == 0 || vecs.at(i+1).at(j) == 0) {
									str_inputs = str_inputs + to_string(0) + ",";
									input.push_back(0);
								} else { 
									input.push_back((vecs.at(i).at(j) - vecs.at(i+1).at(j)) * 10);
									str_inputs = str_inputs + to_string((vecs.at(i).at(j) - vecs.at(i+1).at(j)) * 10) + ",";
								}
							}
						}
						inputs.push_back(input);
					}	
					vecs.erase(vecs.begin());
					PyObject* arg = vectorToList_Float(inputs);
					//여기에서 인자를 위의 inputs를 넣어주고 예측!!!
					if (pModule != NULL)  { 
						pModuleFunc1 = PyObject_GetAttrString(pModule, "action_classification");
		    				if (pModuleFunc1 && PyCallable_Check(pModuleFunc1)) {
							//char *c = const_cast<char*>(a.c_str());
							//PyObject* arg = PyUnicode_FromString(c);
							//npy_intp dims[2] = {16, 54};
        	   					pArgs1 = NULL;
							//PyObject* arg2 = Py_BuildValue("i", 1);
        	   	 				//pReturnVal1 = PyObject_CallObject(pModuleFunc1, arg);
							//PyArrayObject* numpyArray = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, (float*)inputs.data());
							//cout << numpyArray << endl;
							PyObject *pArgs = PyTuple_New( 1 );
							//PyObject *tmp = vectorToList_Float(inputs);
							const char *str1 = str_inputs.c_str();
							PyObject *tmp = PyUnicode_FromString(str1);
							PyTuple_SetItem( pArgs, 0, tmp );	
							pReturnVal1 = PyObject_CallObject(pModuleFunc1, pArgs);
        	  	  			}
        	    				else {
		             				if(PyErr_Occurred())
        					 		  PyErr_Print();
        	    					std::cout << "error: no func\n";
        	    				}
		    				Py_XDECREF(pModuleFunc3);
    		 			}
				}
			}
			// 11. show and save result
			//cout << "people: " << shape[0] << endl;
			//imshow("demo", im); //show image
			//imwrite("output/result.jpg", im);//save result as jpg
			outputVideo << im;
			delete [] heatmap_peaks;
			delete [] heatmap;
			delete [] netin_data;
			cap.read(im);
		}
	}

	else if(strcmp(input_type, "webcam1") == 0) {

		Mat im;
		namedWindow("result", 1); 
		VideoCapture cap(0);  
		cap.set(CAP_PROP_FPS, 20.0);
		vector<vector<float>> vecs;
		vector<float> vec;
		vector<vector<float>> inputs;

		char csvfilename[100];
		csvfilename[0] = '\0';
		strcpy(csvfilename, "csv/data.csv");

		ofstream writeFile(csvfilename);
		if(writeFile.is_open()) {
			cout << "write file open errer" << endl;
		}
		if (!cap.isOpened())  
		{  
			cout << "failed to read video" << endl;
		} 
		Size size = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),(int)cap.get(CAP_PROP_FRAME_HEIGHT));
		VideoWriter outputVideo;
		outputVideo.open("ouput.avi", VideoWriter::fourcc('X', 'V', 'I', 'D'),
			30, size, true);
		cap.read(im);
		imshow("result", im);
		if (im.empty())
		{
			cout << "failed to read image (" << im_path << ")" << endl;
		}

		// 2. initialize net
		int net_inw = 0;
		int net_inh = 0;
		int net_outw = 0;
		int net_outh = 0;
		init_net(cfg_path, weight_path, &net_inw, &net_inh, &net_outw, &net_outh);

		while(!im.empty()) {
			string a = "";
			// 3. resize to net input size, put scaled image on the top left
			float scale = 0.0f;
			Mat netim = create_netsize_im(im, net_inw, net_inh, &scale);

			// 4. normalized to float type
			netim.convertTo(netim, CV_32F, 1 / 256.f, -0.5);

			// 5. split channels
			float *netin_data = new float[net_inw * net_inh * 3]();
			float *netin_data_ptr = netin_data;
			vector<Mat> input_channels;
			for (int i = 0; i < 3; ++i)
			{
				Mat channel(net_inh, net_inw, CV_32FC1, netin_data_ptr);
				input_channels.emplace_back(channel);
				netin_data_ptr += (net_inw * net_inh);
			}
			split(netim, input_channels);

			// 6. feed forward
			double time_begin = getTickCount();
			float *netoutdata = run_net(netin_data);
			double fee_time = (getTickCount() - time_begin) / getTickFrequency() * 1000;
			cout << "forward fee: " << fee_time << "ms" << endl;

			// 7. resize net output back to input size to get heatmap
			float *heatmap = new float[net_inw * net_inh * NET_OUT_CHANNELS];
			for (int i = 0; i < NET_OUT_CHANNELS; ++i)
			{
				Mat netout(net_outh, net_outw, CV_32F, (netoutdata + net_outh*net_outw*i));
				Mat nmsin(net_inh, net_inw, CV_32F, heatmap + net_inh*net_inw*i);
				resize(netout, nmsin, Size(net_inw, net_inh), 0, 0, CV_INTER_CUBIC);
			}

			// 8. get heatmap peaks
			float *heatmap_peaks = new float[3 * (POSE_MAX_PEOPLE+1) * (NET_OUT_CHANNELS-1)];
			find_heatmap_peaks(heatmap, heatmap_peaks, net_inw, net_inh, NET_OUT_CHANNELS, 0.05);

			// 9. link parts
			vector<float> keypoints;
			vector<int> shape;
			connect_bodyparts(keypoints, heatmap, heatmap_peaks, net_inw, net_inh, 9, 0.05, 6, 0.4, shape);

			// 10. draw result
			render_pose_keypoints(im, keypoints, shape, 0.05, scale);
			if(!im.empty()) {
				imshow("result", im); //show image
				waitKey(1);
			}
// 사람이 1명 있을 때 
			if(!keypoints.empty() && keypoints.size() >= 54) {
				string str_inputs = "";
				inputs.clear();
				vecs.push_back(keypoints);
				for(int k=0; k<vecs.size(); k++) 
					standardization(&vecs.at(k));
				if(vecs.size() == 17) {
					for(int i=0; i<vecs.size() - 1; i++) {
						vector<float> input;
						for(int j=0; j<54; j++) {
							
							if(j % 3 == 2) {
								str_inputs = str_inputs + to_string((vecs.at(i).at(j) + vecs.at(i+1).at(j)) / 2.0) + ",";
								input.push_back((vecs.at(i).at(j) + vecs.at(i+1).at(j)) / 2.0);
							} 
							else {
								if(vecs.at(i).at(j) == 0 || vecs.at(i+1).at(j) == 0) {
									str_inputs = str_inputs + to_string(0) + ",";
									input.push_back(0);
								} else { 
									input.push_back((vecs.at(i).at(j) - vecs.at(i+1).at(j)) * 10);
									str_inputs = str_inputs + to_string((vecs.at(i).at(j) - vecs.at(i+1).at(j)) * 10) + ",";
								}
							}
						}
						inputs.push_back(input);
					}	
					vecs.erase(vecs.begin());
					PyObject* arg = vectorToList_Float(inputs);
					//여기에서 인자를 위의 inputs를 넣어주고 예측!!!
					if (pModule != NULL)  { 
						pModuleFunc1 = PyObject_GetAttrString(pModule, "action_classification");
		    				if (pModuleFunc1 && PyCallable_Check(pModuleFunc1)) {
							//char *c = const_cast<char*>(a.c_str());
							//PyObject* arg = PyUnicode_FromString(c);
							//npy_intp dims[2] = {16, 54};
        	   					pArgs1 = NULL;
							//PyObject* arg2 = Py_BuildValue("i", 1);
        	   	 				//pReturnVal1 = PyObject_CallObject(pModuleFunc1, arg);
							//PyArrayObject* numpyArray = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, (float*)inputs.data());
							//cout << numpyArray << endl;
							PyObject *pArgs = PyTuple_New( 1 );
							//PyObject *tmp = vectorToList_Float(inputs);
							const char *str1 = str_inputs.c_str();
							PyObject *tmp = PyUnicode_FromString(str1);
							PyTuple_SetItem( pArgs, 0, tmp );	
							pReturnVal1 = PyObject_CallObject(pModuleFunc1, pArgs);
        	  	  			}
        	    				else {
		             				if(PyErr_Occurred())
        					 		  PyErr_Print();
        	    					std::cout << "error: no func\n";
        	    				}
		    				Py_XDECREF(pModuleFunc3);
    		 			}
				}
			}
			// 11. show and save result
			cout << "people: " << shape[0] << endl;
			//imshow("demo", im); //show image
			//imwrite("output/result.jpg", im);//save result as jpg
			outputVideo << im;
			delete [] heatmap_peaks;
			delete [] heatmap;
			delete [] netin_data;
			cap.read(im);
		}
	}

	else if(strcmp(input_type, "webcam") == 0) {

		Mat im;
		namedWindow("result", 1); 
		VideoCapture cap(0);  
		cap.set(CAP_PROP_FPS, 20.0);
		vector<vector<float>> vecs;
		vector<float> vec;

		char csvfilename[100];
		csvfilename[0] = '\0';
		strcpy(csvfilename, "csv/data.csv");

		ofstream writeFile(csvfilename);
		if(writeFile.is_open()) {
			cout << "write file open errer" << endl;
		}
		if (!cap.isOpened())  
		{  
			cout << "failed to read video" << endl;
		} 
		Size size = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),(int)cap.get(CAP_PROP_FRAME_HEIGHT));
		VideoWriter outputVideo;
		outputVideo.open("ouput.avi", VideoWriter::fourcc('X', 'V', 'I', 'D'),
			30, size, true);
		cap.read(im);
		imshow("result", im);
		if (im.empty())
		{
			cout << "failed to read image (" << im_path << ")" << endl;
		}

		// 2. initialize net
		int net_inw = 0;
		int net_inh = 0;
		int net_outw = 0;
		int net_outh = 0;
		init_net(cfg_path, weight_path, &net_inw, &net_inh, &net_outw, &net_outh);

		while(!im.empty()) {
			string a = "";
			// 3. resize to net input size, put scaled image on the top left
			float scale = 0.0f;
			Mat netim = create_netsize_im(im, net_inw, net_inh, &scale);

			// 4. normalized to float type
			netim.convertTo(netim, CV_32F, 1 / 256.f, -0.5);

			// 5. split channels
			float *netin_data = new float[net_inw * net_inh * 3]();
			float *netin_data_ptr = netin_data;
			vector<Mat> input_channels;
			for (int i = 0; i < 3; ++i)
			{
				Mat channel(net_inh, net_inw, CV_32FC1, netin_data_ptr);
				input_channels.emplace_back(channel);
				netin_data_ptr += (net_inw * net_inh);
			}
			split(netim, input_channels);

			// 6. feed forward
			double time_begin = getTickCount();
			float *netoutdata = run_net(netin_data);
			double fee_time = (getTickCount() - time_begin) / getTickFrequency() * 1000;
			cout << "forward fee: " << fee_time << "ms" << endl;

			// 7. resize net output back to input size to get heatmap
			float *heatmap = new float[net_inw * net_inh * NET_OUT_CHANNELS];
			for (int i = 0; i < NET_OUT_CHANNELS; ++i)
			{
				Mat netout(net_outh, net_outw, CV_32F, (netoutdata + net_outh*net_outw*i));
				Mat nmsin(net_inh, net_inw, CV_32F, heatmap + net_inh*net_inw*i);
				resize(netout, nmsin, Size(net_inw, net_inh), 0, 0, CV_INTER_CUBIC);
			}

			// 8. get heatmap peaks
			float *heatmap_peaks = new float[3 * (POSE_MAX_PEOPLE+1) * (NET_OUT_CHANNELS-1)];
			find_heatmap_peaks(heatmap, heatmap_peaks, net_inw, net_inh, NET_OUT_CHANNELS, 0.05);

			// 9. link parts
			vector<float> keypoints;
			vector<int> shape;
			connect_bodyparts(keypoints, heatmap, heatmap_peaks, net_inw, net_inh, 9, 0.05, 6, 0.4, shape);
			//printf("KeyPoint Count = %d\n", keypoints.size());
			//for(int i=0; i<keypoints.size(); i++) {
			//	printf("%f ", keypoints[i]);
			//}
			//printf("\n");
			// 10. draw result
			render_pose_keypoints(im, keypoints, shape, 0.05, scale);
			if(!im.empty()) {
				imshow("result", im); //show image
				waitKey(1);
			}

			if(shape[0] == 1) {
				vecs.push_back(keypoints);
				//for(int k=0; k<vecs.size(); k++) 
				//	standardization(vecs.at(k));
				if(vecs.size() == 17) {
					vector<float> nomalization;
					for(int i=0; i<vecs.size() - 1; i++) {
						for(int j=0; j<54; j++) {
							if(j % 3 == 2) {
								nomalization.push_back((vecs.at(i).at(j) + vecs.at(i+1).at(j)) / 2.0);
							} 
							else {
								if(vecs.at(i).at(j) == 0 || vecs.at(i+1).at(j) == 0)
									nomalization.push_back(0);
								else 
									nomalization.push_back(vecs.at(i).at(j) - vecs.at(i+1).at(j));
							}
						}
					}
					for(int i=0; i<vecs.size() - 1; i++) {
						for(int j=0; j<54; j++) {
							writeFile << nomalization.at(i) << ",";
						}
					}
					writeFile << endl;	
					vecs.erase(vecs.begin());
				}
			if (pModule != NULL)  { 
				pModuleFunc1 = PyObject_GetAttrString(pModule, "action_classification");
	    			if (pModuleFunc1 && PyCallable_Check(pModuleFunc1)) {
					//char *c = const_cast<char*>(a.c_str());
					//PyObject* arg = PyUnicode_FromString(c);
           				pArgs1 = NULL;
           	 			//pReturnVal1 = PyObject_CallObject(pModuleFunc1, arg);
					pReturnVal1 = PyObject_CallObject(pModuleFunc1, pArgs1);
          	  		}
            			else {
	             			if(PyErr_Occurred())
        			 		  PyErr_Print();
            				std::cout << "error: no func\n";
            			}
	    			Py_XDECREF(pModuleFunc3);
    	 		}
       	 		else {
        			PyErr_Print();
        			std::cout << "\nerror: no module\n";
       			}
			}

			
			// 11. show and save result
			cout << "people: " << shape[0] << endl;
			//imshow("demo", im); //show image
			//imwrite("output/result.jpg", im);//save result as jpg
			outputVideo << im;
			delete [] heatmap_peaks;
			delete [] heatmap;
			delete [] netin_data;
			cap.read(im);
		}
	}

	else if(strcmp(input_type, "rtsp") == 0) {

		Mat im;
		IplImage *frame = NULL;

		CvCapture *capture = cvCaptureFromFile(im_path);  
		if( !capture )    {
			std::cout << "The video file was not found" << std::endl;
			return 0;
		}

		//im = cvQueryFrame(capture);
		frame = cvQueryFrame(capture);
		im = cvarrToMat(frame);

		Size size = Size(im.size());
		VideoWriter outputVideo;
		outputVideo.open("ouput.avi", VideoWriter::fourcc('X', 'V', 'I', 'D'),
			30, size, true);

		if (im.empty())
		{
			cout << "failed to read image (" << im_path << ")" << endl;
		}

		// 2. initialize net
		int net_inw = 0;
		int net_inh = 0;
		int net_outw = 0;
		int net_outh = 0;
		init_net(cfg_path, weight_path, &net_inw, &net_inh, &net_outw, &net_outh);

		while(!im.empty()) {

			// 3. resize to net input size, put scaled image on the top left
			float scale = 0.0f;
			Mat netim = create_netsize_im(im, net_inw, net_inh, &scale);

			// 4. normalized to float type
			netim.convertTo(netim, CV_32F, 1 / 256.f, -0.5);

			// 5. split channels
			float *netin_data = new float[net_inw * net_inh * 3]();
			float *netin_data_ptr = netin_data;
			vector<Mat> input_channels;
			for (int i = 0; i < 3; ++i)
			{
				Mat channel(net_inh, net_inw, CV_32FC1, netin_data_ptr);
				input_channels.emplace_back(channel);
				netin_data_ptr += (net_inw * net_inh);
			}
			split(netim, input_channels);

			// 6. feed forward
			double time_begin = getTickCount();
			float *netoutdata = run_net(netin_data);
			double fee_time = (getTickCount() - time_begin) / getTickFrequency() * 1000;
			cout << "forward fee: " << fee_time << "ms" << endl;

			// 7. resize net output back to input size to get heatmap
			float *heatmap = new float[net_inw * net_inh * NET_OUT_CHANNELS];
			for (int i = 0; i < NET_OUT_CHANNELS; ++i)
			{
				Mat netout(net_outh, net_outw, CV_32F, (netoutdata + net_outh*net_outw*i));
				Mat nmsin(net_inh, net_inw, CV_32F, heatmap + net_inh*net_inw*i);
				resize(netout, nmsin, Size(net_inw, net_inh), 0, 0, CV_INTER_CUBIC);
			}

			// 8. get heatmap peaks
			float *heatmap_peaks = new float[3 * (POSE_MAX_PEOPLE+1) * (NET_OUT_CHANNELS-1)];
			find_heatmap_peaks(heatmap, heatmap_peaks, net_inw, net_inh, NET_OUT_CHANNELS, 0.05);

			// 9. link parts
			vector<float> keypoints;
			vector<int> shape;
			connect_bodyparts(keypoints, heatmap, heatmap_peaks, net_inw, net_inh, 9, 0.05, 6, 0.4, shape);
			printf("KeyPoint Count = %d\n", keypoints.size());
			for(int i=0; i<keypoints.size(); i++) {
				printf("%f ", keypoints[i]);
			} 
			printf("\n");
			// 10. draw result
			render_pose_keypoints(im, keypoints, shape, 0.05, scale);

			// 11. show and save result
			cout << "people: " << shape[0] << endl;
			//imshow("demo", im); //show image
			//imwrite("output/result.jpg", im);//save result as jpg

			delete [] heatmap_peaks;
			delete [] heatmap;
			delete [] netin_data;

			im = cvarrToMat(frame);
			outputVideo << im;
			if(!im.empty()) {
				imshow("result", im); //show image
				waitKey(1);
			}
			frame = cvQueryFrame(capture);
		}
	}


	else if(strcmp(input_type, "train") == 0) {

		Mat im;
		namedWindow("result", 1); 


		char csvfilename[100];
		csvfilename[0] = '\0';
		strcpy(csvfilename, "csv/");
		strcat(csvfilename, im_path);
		mkdir(csvfilename, 0777);
		cout << "mkdir : " << csvfilename << endl;
		strcat(csvfilename, "/");
		strcat(csvfilename, "data.csv");
		cout << "csvfilename : " << csvfilename << endl;
		ofstream writeFile(csvfilename);
		if(writeFile.is_open()) {
			cout << "write file open errer" << endl;
		}
		if(strcmp(train_type, "0") == 0) {

			vector<vector<float>> vecs;
			vector<float> vec;
			vector<vector<float>> nomalvecs;

			DIR *d;
			struct dirent *dir;

			char dir_path[100];
			dir_path[0] = '\0';
			strcpy(dir_path, "data/");
			strcat(dir_path, im_path);

			cout << dir_path << endl;

			d = opendir(dir_path);

			// 2. initialize net
			int net_inw = 0;
			int net_inh = 0;
			int net_outw = 0;
			int net_outh = 0;
			init_net(cfg_path, weight_path, &net_inw, &net_inh, &net_outw, &net_outh);

			if(d) {
				while ((dir = readdir(d)) != NULL) {
					if(strcmp(dir->d_name, ".") == 0 || strcmp(dir->d_name, "..") == 0)
						continue;
					char path[100];
					path[0] = '\0';
					strcpy(path, "data/");
					strcat(path, im_path);
					strcat(path, "/");
					strcat(path, dir->d_name);
					cout << "file path : " << path << endl;
					VideoCapture cap(path); 
					if (!cap.isOpened())  
					{  
						cout << "failed to read video" << endl;
					} 
					Size size = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),(int)cap.get(CAP_PROP_FRAME_HEIGHT));
					cap.read(im);
					imshow("result", im);
					if (im.empty())
					{
						cout << "failed to read image (" << im_path << ")" << endl;
					}

					while(!im.empty()) {
						// 3. resize to net input size, put scaled image on the top left
						float scale = 0.0f;
						Mat netim = create_netsize_im(im, net_inw, net_inh, &scale);

						// 4. normalized to float type
						netim.convertTo(netim, CV_32F, 1 / 256.f, -0.5);

						// 5. split channels
						float *netin_data = new float[net_inw * net_inh * 3]();
						float *netin_data_ptr = netin_data;
						vector<Mat> input_channels;
						for (int i = 0; i < 3; ++i)
						{
							Mat channel(net_inh, net_inw, CV_32FC1, netin_data_ptr);
							input_channels.emplace_back(channel);
							netin_data_ptr += (net_inw * net_inh);
						}
						split(netim, input_channels);

						// 6. feed forward
						double time_begin = getTickCount();
						float *netoutdata = run_net(netin_data);
						double fee_time = (getTickCount() - time_begin) / getTickFrequency() * 1000;
						//cout << "forward fee: " << fee_time << "ms" << endl;

						// 7. resize net output back to input size to get heatmap
						float *heatmap = new float[net_inw * net_inh * NET_OUT_CHANNELS];
						for (int i = 0; i < NET_OUT_CHANNELS; ++i)
						{
							Mat netout(net_outh, net_outw, CV_32F, (netoutdata + net_outh*net_outw*i));
							Mat nmsin(net_inh, net_inw, CV_32F, heatmap + net_inh*net_inw*i);
							resize(netout, nmsin, Size(net_inw, net_inh), 0, 0, CV_INTER_CUBIC);
						}

						// 8. get heatmap peaks
						float *heatmap_peaks = new float[3 * (POSE_MAX_PEOPLE+1) * (NET_OUT_CHANNELS-1)];
						find_heatmap_peaks(heatmap, heatmap_peaks, net_inw, net_inh, NET_OUT_CHANNELS, 0.05);

						// 9. link parts
						vector<float> keypoints;
						vector<int> shape;
						connect_bodyparts(keypoints, heatmap, heatmap_peaks, net_inw, net_inh, 9, 0.05, 6, 0.4, shape);
						//printf("KeyPoint Count = %d\n", keypoints.size());
						//for(int i=0; i<keypoints.size(); i++) {
						//	printf("%f ", keypoints[i]);
						//}
						//printf("\n");
						// 10. draw result
						render_pose_keypoints(im, keypoints, shape, 0.05, scale);
						if(!im.empty()) {
							imshow("result", im); //show image
							waitKey(1);
						}
						// 11. show and save result
						//cout << "people: " << shape[0] << endl;
						if(shape[0] == 1) {
							vecs.push_back(keypoints);
							//for(int k=0; k<vecs.size(); k++) 
							//	standardization(&vecs.at(k));
							if(vecs.size() == 17) {
								vector<float> nomalization;
								for(int i=0; i<vecs.size() - 1; i++) {
									for(int j=0; j<54; j++) {
										if(j % 3 == 2) {
											nomalization.push_back((vecs.at(i).at(j) + vecs.at(i+1).at(j)) / 2.0);
										} 
										else {
											if(vecs.at(i).at(j) == 0 || vecs.at(i+1).at(j) == 0)
												nomalization.push_back(0);
											else 
												nomalization.push_back(vecs.at(i).at(j) - vecs.at(i+1).at(j));
										}
									}
								}
								standardization(&nomalization);
								for(int i=0; i<864; i++) {
									writeFile << nomalization.at(i) << ",";
								}
								writeFile << endl;	
								vecs.erase(vecs.begin());
							}
						}
						//imshow("demo", im); //show image
						//imwrite("output/result.jpg", im);//save result as jpg
						delete [] heatmap_peaks;
						delete [] heatmap;
						delete [] netin_data;
						cap.read(im);
						cap.read(im);
					}
				}
				closedir(d);
				writeFile.close();
			}
		}
	}

	// Follow [1] to get what's going on
        if (pModule != NULL)
        { 
        //pModuleFunc1 = PyObject_GetAttrString(pModule, "action_classification");
	pModuleFunc2 = PyObject_GetAttrString(pModule, "python_close");
	//pModuleFunc3 = PyObject_GetAttrString(pModule, "print3");
            //if (pModuleFunc1 && PyCallable_Check(pModuleFunc1))
            //{
            //pArgs1 = NULL;
            //pReturnVal1 = PyObject_CallObject(pModuleFunc1, pArgs1);
            //}
	    if (pModuleFunc2 && PyCallable_Check(pModuleFunc2))
            {
            pArgs2 = NULL;
            pReturnVal2 = PyObject_CallObject(pModuleFunc2, pArgs2);
            }
	    /*if (pModuleFunc3 && PyCallable_Check(pModuleFunc3))
            {
            pArgs3 = NULL;
            pReturnVal3 = PyObject_CallObject(pModuleFunc3, pArgs3);
            }*/
            else{
                if(PyErr_Occurred())
        	    PyErr_Print();
            std::cout << "error: no func\n";
            }   
	    Py_XDECREF(pModuleFunc1);
	    Py_XDECREF(pModuleFunc2);
            Py_DECREF(pModule);
    	    }
        else
        {
        PyErr_Print();
        std::cout << "\nerror: no module\n";
        }


	

	return 0;
}
