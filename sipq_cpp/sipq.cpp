//������ ��Ƽ�����幮���� �ᱹ ��뷮 �����Ϳ� ���Ͽ� ĳ����Ʈ���� ���� ����ð��� �޶����� ������ �Ǵܵȴ�.
//�� �˰��� Ư���� ���� �����͸� ���ʿ��� �������� �����ʿ��� �������� �����ϱ� ��Ʊ� ������ ĳ�� ��Ʈ����
//���� ���̰� ��Į�� �������� ���ʿ��� ���ʴ�� �������� ������ ĳ�� ��Ʈ���� ���� ���̴� �׷��� ��Į�� �� ������ 
//���̴�.�׸��� ��������� 255���� ���� �� ������ ������ �õ� Ƚ���� �� ���� ������ ĳ�� ��Ʈ���� ���������̴�.
//pmjs�� ���������� �������� ������ �����Ϸ��� �����ϱ� ������ ��Ʈ���� ���� �ִ� ������ �����̴�.

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include <xmmintrin.h>
#include <nmmintrin.h>
#include <emmintrin.h>
#include <thread>
#include <limits.h>
#include <random>

using namespace std;

#define NUM_DATA 100000000 //������ ����
#define MAX_VALUE INT_MAX - 1 //�ִ밪�� INT_MAX ���� 1��ŭ �۾ƾ� �Ѵ� (�е������� �ִ밪�� �־�� ��)

#define STKSIZE 100'000 //stack size

#define SIMD_SIZE_BIT 256 //the size of SIMD register
#define NUM_SIMD_DATA SIMD_SIZE_BIT / 32 //SIMD�� ������ �� �ִ� 32��Ʈ ������ ����
#define NUM_SIMD_SHIFT 3 //SIMD �������Ϳ� �̸� ������ ���� ����

#define RDMAX 32'767

#define NUM_THREADS 4 //������ ���� (�⺻�� : 16)
#define NUM_RANGES 1024 //���� ���� (�⺻�� : 1024)
#define NUM_SECTIONS 4096 //���� ���� (�⺻�� : 4096)
#define NUM_SAMPLES 512 //���ø� ���� (�⺻�� : 512)

#define SORTSIMD 1 //1: SIMD ���� , 0: ��Į�� ����

#define LT(a,b) arr[a]<arr[b]
#define MED3(a,b,c) (LT(a,b)?((LT(b,c))?b:(LT(a,c)?c:a)):(LT(c,b)?b:(LT(a,c)?a:c)))
#define SWAP(a,b) {int temp = arr[a];arr[a]=arr[b];arr[b]=temp;}

/* INFO: SIMD �������� 3���� �����ư��� ����ϱ� ���� ���Ǵ� ����ü */
struct list_node_t {
	int data;
	struct list_node_t* rlink;
};

int maxValue; //�����͵� �� �ִ밪
int minValue; //�����͵� �� �ּҰ�

#define SMALLLOC {\
	for (int i = m + (cnt << NUM_SIMD_SHIFT); i < n;) {\
		if (arr[i] > pivot) {\
			n--;\
			SWAP(n, i);\
		}\
		else if (arr[i] < pivot)\
			arr[m++] = arr[i++];\
		else {\
			i++;\
		}\
	}\
}

#define LOCMEDIUM {/*lPos�� rPos ���̿�(���� �ѹ��� �˻���� ����)�����Ͱ� �����ϸ� �̵��� ���� �Ǻ��� ���Ͽ� ������ m, ���ų� ũ�� n�� ���� */\
	pureRPos = lrPick[1] + SDMinus;/*rPos�� ���� ������ ������ ��ġ�� ���*/\
	for (int i = lrPick[0]; i <= pureRPos;) {\
		if (arr[i] > pivot) {/*�����Ͱ� �Ǻ����� ������*/\
			SWAP(pureRPos,i);\
			arr[--n] = arr[pureRPos--]; \
		}\
		else if(arr[i] < pivot)\
			arr[m++] = arr[i++];/*m�� �����͸� ����*/\
		else{\
			i++;\
		}\
	}\
}

#define SIMDOP {\
	simdResult = _mm256_cmpgt_epi32(simdPivot, simdBuffer[bfIdx]);\
	res_f = _mm256_castsi256_ps(simdResult);\
	mask = _mm256_movemask_ps(res_f);\
	p = _mm256_permutevar8x32_epi32(simdBuffer[bfIdx], shuffle_mask[mask]);\
	_mm256_storeu_si256((__m256i*)&arr[m], p);\
	m += _mm_popcnt_u32(mask);\
	simdResult = _mm256_cmpgt_epi32(simdBuffer[bfIdx], simdPivot);\
	res_f = _mm256_castsi256_ps(simdResult);\
	mask = _mm256_movemask_ps(res_f);\
	p = _mm256_permutevar8x32_epi32(simdBuffer[bfIdx], rshuffle_mask[mask]);\
	_mm256_storeu_si256((__m256i*)&arr[n - NUM_SIMD_DATA], p);\
	n -= _mm_popcnt_u32(mask);\
}

#define RESTSIMDOP {\
	simdResult = _mm256_cmpgt_epi32(simdPivot, simdBuffer[bfIdx]); \
	res_f = _mm256_castsi256_ps(simdResult);\
	mask = _mm256_movemask_ps(res_f);\
	p = _mm256_permutevar8x32_epi32(simdBuffer[bfIdx], shuffle_mask[mask]);\
	_mm256_storeu_si256((__m256i*)&arr[m], p);\
	m += _mm_popcnt_u32(mask);\
	simdResult = _mm256_cmpgt_epi32(simdBuffer[bfIdx], simdPivot);\
	res_f = _mm256_castsi256_ps(simdResult);\
	mask = _mm256_movemask_ps(res_f);\
	p = _mm256_permutevar8x32_epi32(simdBuffer[bfIdx], shuffle_mask[mask]);\
	_mm256_storeu_si256((__m256i*)&restArr, p);\
	restCnt = _mm_popcnt_u32(mask);\
	memcpy(&arr[n - restCnt], restArr, restCnt << 2);\
	n -= restCnt;\
}

int nrData{ 0 }; //���� �� ������ ����
int originNrData{ 0 }; //���� �� ������ ����

__m256i shuffle_mask[256];
__m256i rshuffle_mask[256];
int songShuffle_mask[256] = { 0, };
clock_t begin, end;

/* INFO: ������ ������ Ŭ���� */
class random_generator {
private:
	//INFO: �õ尪�� ��� ���� ����
	std::random_device rd;
	//INFO: �������� ���� �ʱ�ȭ
	mt19937 gen;
	//INFO: �յ��ϰ� ��Ÿ���� �������� �����ϱ� ���� �յ� ���� ����
	std::uniform_int_distribution<int> dis;
public:
	random_generator(int minValue, int maxValue) : gen(rd()), dis(minValue, maxValue) {
		printf("Init random_generator ok.\n");
	}

	int getRandomValue() {
		return dis(gen);
	}
};

random_generator random(0, MAX_VALUE);

/* INFO: SIMD���� Ȱ���� ������̺� ���� */
void initLookupTable256() { 
	int bitCheck;
	int bitCount;
	int remainSpot;
	for (int i = 0; i < SIMD_SIZE_BIT; i++) {
		bitCount = 0;
		for (int k = 0; k < NUM_SIMD_DATA; k++) {
			bitCheck = (i >> k) & 1;
			if (bitCheck == 1) {
				shuffle_mask[i].m256i_i32[bitCount++] = k;
			}
		}
		remainSpot = NUM_SIMD_DATA - bitCount;
		for (int j = 0; j < remainSpot; j++) {
			shuffle_mask[i].m256i_i32[bitCount + j] = -1;
		}
	}
	//rshuffle_mask�� �Ǻ��� ���ų� ū �����͵��� ���� �迭�� ������ �� �����͵��� ������ ���� ���̱� ���� ���̺��� ���� ������(��: -1 -1 -1 1 0)
	for (int i = 0; i < SIMD_SIZE_BIT; i++) {
		bitCount = NUM_SIMD_DATA - 1;
		for (int k = 0; k < NUM_SIMD_DATA; k++) {
			bitCheck = (i >> k) & 1;
			if (bitCheck == 1) {
				rshuffle_mask[i].m256i_i32[bitCount--] = k;
			}
		}
		remainSpot = bitCount;
		for (int j = 0; j <= remainSpot; j++) {
			rshuffle_mask[i].m256i_i32[j] = 0;
		}
	}
	songShuffle_mask[0] = 0;
	for (int i = 1; i < SIMD_SIZE_BIT; i++) {
		for (int k = NUM_SIMD_DATA - 1; k >= NUM_SIMD_DATA; k--) {
			bitCheck = (i >> k) & 1;
			if (bitCheck == 1) {
				songShuffle_mask[i] = NUM_SIMD_DATA - 1 - k;
				break;
			}
		}
	}
}

/* INFO: ������ �迭�� ���� ���� �����ϰ� �ʱ�ȭ */
void initSingleArrayData(int** arr, int& nrData, int& originNrData) {
	int cnt;
	int balanceSize;

	nrData = NUM_DATA;
	originNrData = nrData;
	balanceSize = nrData;

	cnt = (nrData % NUM_RANGES);
	if (cnt != 0) cnt = NUM_RANGES - cnt;
	balanceSize += cnt;

	(*arr) = (int*)malloc(sizeof(int*) * balanceSize);
	for (int i = 0; i < nrData; i++) {
		(*arr)[i] = random.getRandomValue();
	}

	maxValue = (*arr)[0];
	minValue = INT_MAX;
	for (int i = 1; i < originNrData; i++) {
		if (maxValue < (*arr)[i]) maxValue = (*arr)[i];
		if (minValue > (*arr)[i]) minValue = (*arr)[i];
	}
	printf("minValue : %d, maxValue : %d\n", minValue, maxValue);

	for (int i = nrData; i < balanceSize; i++) {
		(*arr)[i] = maxValue + 1;
	}
	nrData = balanceSize;
}

/* INFO: ��Į�� ����� ������ */
void scalarQuickSort(int* arr, int nrData) {
	int top = 0;
	int stack[STKSIZE];
	int p, t;
	int i, j;
	int low, high;
	int n;
	int a, b, c, d;
	int tmp;
	low = 0;
	high = nrData - 1;
	top = 0;
	stack[top++] = high;
	stack[top++] = low;
	while (top != 0) {
		low = (top != 0) ? stack[--top] : 0;
		high = (top != 0) ? stack[--top] : 0;
		if (high > low) {
			n = high - low + 1;
			b = (low + high) >> 1;//middle element
			if (n > 7) {
				a = low;
				c = high;
				if (n > 40) {
					d = n >> 3;
					a = MED3(a, a + d, a + 2 * d);
					b = MED3(b - d, b, b + d);
					c = MED3(c - 2 * d, c - d, c);
				}
				b = MED3(a, b, c);
			}
			if (b != high) {
				tmp = arr[high]; arr[high] = arr[b]; arr[b] = tmp;
			}
			p = arr[high];
			i = low - 1;
			j = high;
			while (1) {
				while ((i < j) && (arr[++i] < p));
				while ((i < j) && (arr[--j] > p));
				if (i >= j)
					break;
				t = arr[i];
				arr[i] = arr[j];
				arr[j] = t;
			}

			t = arr[i];
			arr[i] = arr[high];
			arr[high] = t;

			stack[top++] = high; //���ο� ���� low�� high�� �Է�
			stack[top++] = i + 1;
			stack[top++] = i - 1;
			stack[top++] = low;
		}
	}
}

/* INFO: SIMD 256bit ����� ������ */
void SIMD256QuickSort(int* arr, int nrData) {
	list_node_t bufferList[3];
	bufferList[0].rlink = &bufferList[1];
	bufferList[0].data = 0;
	bufferList[1].rlink = &bufferList[2];
	bufferList[1].data = 1;
	bufferList[2].rlink = &bufferList[0];
	bufferList[2].data = 2;

	int top = 0;
	int stack[STKSIZE];
	int low, high;
	top = 0;
	low = 0;
	high = nrData - 1;
	stack[top++] = high;
	stack[top++] = low;

	int mask = 0;
	int lPos, rPos;
	int lrPick[2];
	int lrMove[2] = { NUM_SIMD_DATA,-NUM_SIMD_DATA };
	int newPivot = 0;
	int ibCnt, jbCnt;
	int m, n;
	int mStart;
	int cnt = 0;
	int temp;
	int SDMinus = NUM_SIMD_DATA - 1;

	int endBound;
	int gap;
	int key, pos;
	int pureLPos, pureRPos;
	int moveDist;
	int cmp;
	int nMinus;
	int bfIdx;
	int SDTRP = NUM_SIMD_DATA * 3;
	int SDDBL = NUM_SIMD_DATA * 2;
	int db;
	int pivot;

	int mc = 0;//�Ǻ��� ���� ���� ����
	int sameMask;
	int restArr[NUM_SIMD_DATA];
	int restCnt;

	int a, b, c, d;
	__m256i p;
	__m256 res_f;
	__m256i simdResult;
	__m256i simdSame;
	__m256i simdComp;
	__m256i simdPivot;
	__m256i simdBuffer[3];

	while (top != 0) { //���� ���ÿ� �����Ͱ� �ִٸ�
		low = (top != 0) ? stack[--top] : 0;
		high = (top != 0) ? stack[--top] : 0;
		if (high > low) {//high�� low���� ũ�ٸ� 
			mc = high - low; //low�� ������ ����
			b = (low + high) >> 1;//middle element
			m = low + 1;//�Ǻ����� ���� �����͸� ������ ���� ��ġ
			n = high + 1;//�Ǻ��� ���ų� ū �����͸� ������ ���� ��ġ

			if (SDTRP > mc) {

				b = MED3(low, b, high);
				SWAP(low, b);
				pivot = arr[low];
				cnt = (mc >> NUM_SIMD_SHIFT);
				if (cnt == 0) {
					SMALLLOC;
				}
				else {
					simdPivot = _mm256_set1_epi32(pivot);//�Ǻ� ���� SIMD�������Ϳ� �ߺ�����
					for (int i = 0; i < cnt; i++) {
						simdBuffer[i] = _mm256_loadu_si256((__m256i*) & arr[m + (i << NUM_SIMD_SHIFT)]);
					}
					SMALLLOC;
					if (cnt == 1) {
						bfIdx = 0;
						RESTSIMDOP;
					}
					else {
						bfIdx = 1;
						SIMDOP;
						bfIdx = 0;
						RESTSIMDOP;
					}
				}
			}
			else {
				d = mc >> 3;
				db = d << 1;
				a = MED3(low, low + d, low + db);
				b = MED3(b - d, b, b + d);
				c = MED3(high - db, high - d, high);
				b = MED3(a, b, c);
				SWAP(low, b);
				pivot = arr[low];
				lrPick[0] = m;//lPos
				lrPick[1] = high - SDMinus; //rpos
				simdPivot = _mm256_set1_epi32(pivot);//�Ǻ� ���� SIMD�������Ϳ� �ߺ�����

				simdBuffer[0] = _mm256_loadu_si256((__m256i*) & arr[lrPick[0]]); //���� �����͵��� ���ʴ�� ����
				lrPick[0] += NUM_SIMD_DATA;
				simdBuffer[1] = _mm256_loadu_si256((__m256i*) & arr[lrPick[1]]); //���� �����͵��� ���ʴ�� ����
				lrPick[1] -= NUM_SIMD_DATA;
				simdBuffer[2] = _mm256_loadu_si256((__m256i*) & arr[lrPick[1]]); //���� �����͵��� ���ʴ�� ����
				lrPick[1] -= NUM_SIMD_DATA;
				bfIdx = 0;

				while (1) {
					SIMDOP;
					if (lrPick[0] <= lrPick[1]) {
						cmp = (lrPick[0] - m) >= NUM_SIMD_DATA; //���� ������ ������ ä��� ������ �������� ä��. ���ʴ� ������ ���� ����
						simdBuffer[bfIdx] = _mm256_loadu_si256((__m256i*) & arr[lrPick[cmp]]); //���� �����͵��� ���ʴ�� ����
						lrPick[cmp] += lrMove[cmp];
						bfIdx = bufferList[bfIdx].rlink->data;
					}
					else {
						LOCMEDIUM;
						bfIdx = bufferList[bfIdx].rlink->data;
						SIMDOP;
						bfIdx = bufferList[bfIdx].rlink->data;
						RESTSIMDOP;
						break;
					}
				}
			}

			newPivot = m - 1;//�Ǻ� ��ġ ����
			SWAP(newPivot, low);
			for (int i = m; i < n; i++) arr[i] = arr[newPivot];

			stack[top++] = high; //���ο� ���� low�� high�� �Է�
			stack[top++] = n;
			stack[top++] = newPivot - 1;
			stack[top++] = low;
		}
	}
}

/* INFO: �־��� �迭�� ���� ������, ������, ū������ �з� */
void rangePartition(int* arr, int nrData, int* rngBndry, int* pttIdx, int start) {
	list_node_t bufferList[3];
	bufferList[0].rlink = &bufferList[1];
	bufferList[0].data = 0;
	bufferList[1].rlink = &bufferList[2];
	bufferList[1].data = 1;
	bufferList[2].rlink = &bufferList[0];
	bufferList[2].data = 2;

	int top = 0;
	int stack[STKSIZE];
	int pvtTop = 0;
	int pvtStack[NUM_RANGES];
	pvtStack[pvtTop++] = NUM_RANGES - 1;
	pvtStack[pvtTop++] = 0;
	int low, high;
	low = 0;
	high = nrData - 1;
	stack[top++] = high;
	stack[top++] = low;

	int mask = 0;
	int lPos, rPos;
	int lrPick[2];
	int lrMove[2] = { NUM_SIMD_DATA,-NUM_SIMD_DATA };
	int newPivot = 0;
	int ibCnt, jbCnt;
	int m, n;
	int mStart;
	int cnt = 0;
	int temp;
	int SDMinus = NUM_SIMD_DATA - 1;

	int endBound;
	int gap;
	int key, pos;
	int pureLPos, pureRPos;
	int moveDist;
	int cmp;
	int nMinus;
	int bfIdx;
	int SDTRP = NUM_SIMD_DATA * 3;
	int SDDBL = NUM_SIMD_DATA * 2;
	int db;

	int mc = 0;//�Ǻ��� ���� ���� ����
	int sameMask;

	int restArr[NUM_SIMD_DATA];
	int restCnt;

	int a, b, c, d;
	__m256i p;
	__m256 res_f;
	__m256i simdResult;
	__m256i simdSame;
	__m256i simdComp;
	__m256i simdPivot;
	__m256i simdBuffer[3];
	n = 0;
	int idxCnt = 0;
	int pivot;
	int pvtLow, pvtHigh;
	int pvtIdx;

	while (top != 0) { //���� ���ÿ� �����Ͱ� �ִٸ�
		low = (top != 0) ? stack[--top] : 0;
		high = (top != 0) ? stack[--top] : 0;
		pvtLow = (pvtTop != 0) ? pvtStack[--pvtTop] : 0;
		pvtHigh = (pvtTop != 0) ? pvtStack[--pvtTop] : 0;
		if ((high > low) && (pvtHigh >= pvtLow)) {//high�� low���� ũ�ٸ� 
			pvtIdx = (pvtHigh + pvtLow) >> 1;
			pivot = rngBndry[pvtIdx]; //2�� ����

			mc = high - low + 1; //low�� ������ ����
			m = low;//�Ǻ����� ���� �����͸� ������ ���� ��ġ
			n = high + 1;//�Ǻ��� ���ų� ū �����͸� ������ ���� ��ġ
			if (SDTRP > mc) {
				cnt = (mc >> NUM_SIMD_SHIFT);
				if (cnt == 0) {
					SMALLLOC;
				}
				else {
					simdPivot = _mm256_set1_epi32(pivot);//�Ǻ� ���� SIMD�������Ϳ� �ߺ�����
					for (int i = 0; i < cnt; i++) {
						simdBuffer[i] = _mm256_loadu_si256((__m256i*) & arr[m + (i << NUM_SIMD_SHIFT)]);
					}
					SMALLLOC;
					if (cnt == 1) {
						bfIdx = 0;
						RESTSIMDOP;
					}
					else {
						bfIdx = 1;
						SIMDOP;
						bfIdx = 0;
						RESTSIMDOP;
					}
				}
			}
			else {
				lrPick[0] = m;//lPos
				lrPick[1] = high - SDMinus; //rpos
				simdPivot = _mm256_set1_epi32(pivot);//�Ǻ� ���� SIMD�������Ϳ� �ߺ�����
				simdBuffer[0] = _mm256_loadu_si256((__m256i*) & arr[lrPick[0]]); //���� �����͵��� ���ʴ�� ����
				lrPick[0] += NUM_SIMD_DATA;
				simdBuffer[1] = _mm256_loadu_si256((__m256i*) & arr[lrPick[1]]); //���� �����͵��� ���ʴ�� ����
				lrPick[1] -= NUM_SIMD_DATA;
				simdBuffer[2] = _mm256_loadu_si256((__m256i*) & arr[lrPick[1]]); //���� �����͵��� ���ʴ�� ����
				lrPick[1] -= NUM_SIMD_DATA;
				bfIdx = 0;

				while (1) {
					SIMDOP;
					if (lrPick[0] <= lrPick[1]) {
						cmp = (lrPick[0] - m) >= NUM_SIMD_DATA; //���� ������ ������ ä��� ������ �������� ä��. ���ʴ� ������ ���� ����
						simdBuffer[bfIdx] = _mm256_loadu_si256((__m256i*) & arr[lrPick[cmp]]); //���� �����͵��� ���ʴ�� ����
						lrPick[cmp] += lrMove[cmp];
						bfIdx = bufferList[bfIdx].rlink->data;
					}
					else {
						LOCMEDIUM;
						bfIdx = bufferList[bfIdx].rlink->data;
						SIMDOP;
						bfIdx = bufferList[bfIdx].rlink->data;
						RESTSIMDOP;
						break;
					}
				}
			}
			for (int i = m; i < n; i++) arr[i] = pivot;

			stack[top++] = high; //���ο� ���� low�� high�� �Է�
			stack[top++] = n;
			stack[top++] = m - 1;
			stack[top++] = low;
			pvtStack[pvtTop++] = pvtHigh;
			pvtStack[pvtTop++] = pvtIdx + 1;
			pvtStack[pvtTop++] = pvtIdx - 1;
			pvtStack[pvtTop++] = pvtLow;

			if (pvtIdx + 1 != NUM_RANGES) pttIdx[pvtIdx + 1] = start + m;
		}
	}
}

/* INFO: 1�������� KDE�� ���� ������ ������ ���� �����͸� �з��� */
void phase1_threadSort(int* arr, int nrData, int thrNum, int* rngBndry, int** pttIdx) {
	int interval, low, high;
	int ptNum, thrCnt, thrIdxTerm;
	int cnt;
	interval = nrData / NUM_RANGES;
	thrCnt = NUM_RANGES / NUM_THREADS;
	thrIdxTerm = interval * NUM_THREADS;
	int startPTPos = thrNum * interval;
	int lastIdx = 0;
	for (int i = 0; i < thrCnt; i++) {
		low = startPTPos + thrIdxTerm * i;
		high = low + interval - 1;
		ptNum = thrNum + (NUM_THREADS * i);
		pttIdx[ptNum][0] = low;
		pttIdx[ptNum][NUM_RANGES] = high + 1;
		rangePartition(&arr[low], interval, rngBndry, pttIdx[ptNum], low);
		for (int j = 1; j < NUM_RANGES; j++) {
			pttIdx[ptNum][j] = (pttIdx[ptNum][j] == -1) ? pttIdx[ptNum][j - 1] : pttIdx[ptNum][j];
		}
		//printf("pttIdx[%d][1]:%d, pttIdx[%d][%d]:%d\n", ptNum, pttIdx[ptNum][1], ptNum, NUM_RANGES - 1, pttIdx[ptNum][NUM_RANGES - 1]);
	}
}

/* INFO: �� �����尡 �־��� ���̷ε忡 ���� ���� ���� */
void threadPartition(int* arr, int* newArr, int** pttIdx, int* rngPosArr, int thrNum) {
	int RGPos = 0;
	int nrRng = 0;
	int thrCnt = NUM_RANGES / NUM_THREADS;
	int rngNum = 0;
	for (int i = 0; i < thrCnt; i++) {
		rngNum = thrNum + (NUM_THREADS * i);
		RGPos = 0;
		nrRng = 0;
		for (int j = 0; j < NUM_RANGES; j++) {
			nrRng = pttIdx[j][rngNum + 1] - pttIdx[j][rngNum];
			memcpy(&newArr[rngPosArr[rngNum] + RGPos], &arr[pttIdx[j][rngNum]], sizeof(int) * nrRng);
			RGPos += nrRng;
		}
#if(SORTSIMD == 1)
		SIMD256QuickSort(&newArr[rngPosArr[rngNum]], RGPos);
#else
		scalarQuickSort(&newArr[rngPosArr[rngNum]], RGPos);
#endif
	}
}

/* INFO: KDE ����� �̿��Ͽ� ���� �������� ��� */
void calculateKDE(int* arr, double sectionDensity[], int nrData) {
	double sampleArr[NUM_SAMPLES];	//���ø� ������ arr
	double stdDeviation = 0.0;//ǥ������
	double avg = 0.0;//���
	double bandwidth = 0.0;//�뿪��
	int randomPos;

	int scaleFactor;
	if (nrData > RDMAX) scaleFactor = (nrData - RDMAX) / RDMAX;
	else scaleFactor = 0;
	for (int i = 0; i < NUM_SAMPLES; i++) { //���� ���� �����ϴ� ���� sampleArr�� ������ arr ���� ����
		randomPos = rand() * scaleFactor + rand() % nrData;
		sampleArr[i] = (double)(arr[randomPos]);
		avg += sampleArr[i];
	}
	avg /= NUM_SAMPLES;//��հ�
	double temp;
	for (int i = 0; i < NUM_SAMPLES; i++) {
		temp = sampleArr[i] - avg;
		stdDeviation += (temp * temp);
	}
	stdDeviation /= NUM_SAMPLES;
	stdDeviation = sqrt(stdDeviation);//ǥ������
	bandwidth = 1.06 * stdDeviation * pow(NUM_SAMPLES, -0.2);//�뿪���� ��Ű�ǵ�Ƹ� �����Ͽ� ������. ���߿� ������ �� ���氡��
	int intervalOfSection = maxValue / NUM_SECTIONS;
	double hSMPgop = (NUM_SAMPLES * bandwidth);
	temp = 0.0;

	for (int i = 0; i < NUM_SECTIONS; i++) {//������
		for (int j = 0; j < NUM_SAMPLES; j++) { //���ü�
			temp = ((double)(i * intervalOfSection) - sampleArr[j]) / bandwidth; //������� u ��
			temp *= -0.5 * temp;
			sectionDensity[i] += 399000000.0 * pow(2.7183, temp); //���� �ʹ� �۰� ���ͼ� ����� �ø�
		}
		sectionDensity[i] /= hSMPgop;
	}
}

/* INFO: ���� �迭�� ���� �յ���Ϻй� ��� */
void distributeWorkloadWithSingleArray(int* arr, int* rngBndry, int nrData) {
	//INFO: �迭���� KDE�� ���� ������ �� ������ �е�
	double* sectionDensity = (double*)malloc(sizeof(double) * NUM_SECTIONS);
	//INFO: �е� ����
	double totalDensity = 0.0;

	for (int i = 0; i < NUM_SECTIONS; i++) {
		sectionDensity[i] = 0.0;
	}

	calculateKDE(arr, sectionDensity, nrData);

	for (int i = 0; i < NUM_SECTIONS; i++) {
		totalDensity += sectionDensity[i];
	}

	double abThrhld = totalDensity / (double)NUM_RANGES;
	double reltThrhld = 0.0f;
	double restThr = 0.0f;
	double cdf = 0.0f;
	int intervalOfSection = maxValue / NUM_SECTIONS;
	int cnt = 0;
	reltThrhld = abThrhld - restThr;

	for (int i = 0; i < NUM_SECTIONS; i++) {
		cdf += sectionDensity[i];
		if (cdf >= reltThrhld) {
			rngBndry[cnt++] = intervalOfSection * (i + 1);
			restThr = cdf - reltThrhld;
			reltThrhld = abThrhld - restThr;
			cdf = 0.0;
			if (cnt == NUM_RANGES) break;
		}
	}
	while (cnt < NUM_RANGES) {
		rngBndry[cnt++] = maxValue;
	}
	free(sectionDensity);
}

/* INFO: ���� ������ �����ϴ� �Լ� */
void SIPQ(int** arr, int nrData, int* rngPosArr, int* rngBndry) {
	int* newArr = (int*)malloc(sizeof(int) * nrData);
	clock_t begin, end;
	int* pttIdx[NUM_RANGES];
	for (int i = 0; i < NUM_RANGES; i++) {
		pttIdx[i] = (int*)malloc(sizeof(int) * (NUM_RANGES + 1));
		for (int j = 0; j < NUM_RANGES + 1; j++) {
			pttIdx[i][j] = -1;
		}
	}

	thread* thr[NUM_THREADS];

	rngPosArr[0] = 0;

	///////////////////////////////////////////
	for (int i = 0; i < NUM_THREADS; i++) {
		thr[i] = new thread(&phase1_threadSort, (*arr), nrData, i, rngBndry, pttIdx);
	}
	for (int i = 0; i < NUM_THREADS; i++) {
		thr[i]->join();
	}

	for (int i = 0; i < NUM_THREADS; i++) {
		delete thr[i];
	}
	///////////////////////////////////////////

	int ptNrData = 0;
	for (int i = 0; i < NUM_RANGES; i++) {
		ptNrData = 0;
		for (int j = 0; j < NUM_RANGES; j++) {
			ptNrData += pttIdx[j][i + 1] - pttIdx[j][i];
		}
		rngPosArr[i + 1] = rngPosArr[i] + ptNrData;
	}

	///////////////////////////////////////////
	for (int i = 0; i < NUM_THREADS; i++) {
		thr[i] = new thread(&threadPartition, (*arr), newArr, pttIdx, rngPosArr, i);
	}
	for (int i = 0; i < NUM_THREADS; i++) {
		thr[i]->join();
	}
	for (int i = 0; i < NUM_THREADS; i++) {
		delete thr[i];
	}
	///////////////////////////////////////////

	int* originArr = (*arr);
	(*arr) = newArr;

	for (int i = 0; i < NUM_RANGES; i++) {
		free(pttIdx[i]);
	}
	free(originArr);
}

void main() {
	clock_t begin, end;

	initLookupTable256();

	int* vArr = nullptr;
	double totalTime = 0.0;

	initSingleArrayData(&vArr, nrData, originNrData);
	printf("data size - nrData : %d\n", nrData);

	int* rngPosArrR = (int*)malloc(sizeof(int) * (NUM_RANGES + 1));
	int* rngBndry = (int*)malloc(sizeof(int) * NUM_RANGES);
	for (int i = 0; i < NUM_RANGES; i++) rngBndry[i] = -1;

	printf("Start quicksort\n");

	begin = clock();
	distributeWorkloadWithSingleArray(vArr, rngBndry, nrData);
	end = clock();
	printf("��Ƽ������ �۾� �й� ��� �ð� : %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);
	totalTime += (double)(end - begin);

	begin = clock();
	SIPQ(&vArr, nrData, rngPosArrR, rngBndry);
	end = clock();
	printf("��Ƽ������ ���� ���� �ð� : %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);
	totalTime += (double)(end - begin);

	//INFO: �Ϻ� �����͸� ǥ�� (Ȯ�ο�)
	int valInterval = nrData > 10000 ? nrData / 1000 : 1;
	for (int i = 0; i < nrData; i+= valInterval) {
		printf("%d ", vArr[i]);
	}
	printf("\n");

	printf("firstValue : %d, lastValue : %d\n", vArr[0], vArr[originNrData - 1]);

	//INFO : SIMD�� ���� �߰������� �Ҵ�� ������ ������ ���� ũ��� ��ȯ
	nrData = originNrData;

	printf("Finish quicksort\n");

	printf("��ü ���� �ð� : %lf\n", (double)(totalTime / CLOCKS_PER_SEC));

	free(rngPosArrR);
	free(vArr);
	free(rngBndry);
}