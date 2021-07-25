//퀵정렬 멀티스레드문제는 결국 대용량 데이터에 대하여 캐시히트율에 따라 수행시간이 달라지는 것으로 판단된다.
//내 알고리즘 특성상 비교할 데이터를 왼쪽에서 가져올지 오른쪽에서 가져올지 예측하기 어렵기 때문에 캐시 히트율이
//낮은 것이고 스칼라 퀵정렬은 왼쪽에서 차례대로 가져오기 때문에 캐시 히트율이 높은 것이다 그래서 스칼라가 더 빨랐던 
//것이다.그리고 스레드수를 255개로 했을 때 빨랐던 이유는 시도 횟수가 더 많기 때문에 캐시 히트율이 높아진것이다.
//pmjs는 순차적으로 가져오기 때문에 컴파일러가 예측하기 쉬워서 히트율이 높아 최대 성능을 낸것이다.

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

#define NUM_DATA 100000000 //데이터 개수
#define MAX_VALUE INT_MAX - 1 //최대값은 INT_MAX 보다 1만큼 작아야 한다 (패딩공간에 최대값을 넣어야 함)

#define STKSIZE 100'000 //stack size

#define SIMD_SIZE_BIT 256 //the size of SIMD register
#define NUM_SIMD_DATA SIMD_SIZE_BIT / 32 //SIMD에 적재할 수 있는 32비트 정수의 개수
#define NUM_SIMD_SHIFT 3 //SIMD 레지스터에 미리 적재할 묶음 개수

#define RDMAX 32'767

#define NUM_THREADS 4 //스레드 개수 (기본값 : 16)
#define NUM_RANGES 1024 //범위 개수 (기본값 : 1024)
#define NUM_SECTIONS 4096 //구간 개수 (기본값 : 4096)
#define NUM_SAMPLES 512 //샘플링 개수 (기본값 : 512)

#define SORTSIMD 1 //1: SIMD 정렬 , 0: 스칼라 정렬

#define LT(a,b) arr[a]<arr[b]
#define MED3(a,b,c) (LT(a,b)?((LT(b,c))?b:(LT(a,c)?c:a)):(LT(c,b)?b:(LT(a,c)?a:c)))
#define SWAP(a,b) {int temp = arr[a];arr[a]=arr[b];arr[b]=temp;}

/* INFO: SIMD 레지스터 3개를 번갈아가며 사용하기 위해 사용되는 구조체 */
struct list_node_t {
	int data;
	struct list_node_t* rlink;
};

int maxValue; //데이터들 중 최대값
int minValue; //데이터들 중 최소값

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

#define LOCMEDIUM {/*lPos와 rPos 사이에(아직 한번도 검사되지 않은)데이터가 존재하면 이들을 지금 피봇과 비교하여 작으면 m, 같거나 크면 n에 복사 */\
	pureRPos = lrPick[1] + SDMinus;/*rPos의 가장 오른쪽 데이터 위치를 계산*/\
	for (int i = lrPick[0]; i <= pureRPos;) {\
		if (arr[i] > pivot) {/*데이터가 피봇보다 작으면*/\
			SWAP(pureRPos,i);\
			arr[--n] = arr[pureRPos--]; \
		}\
		else if(arr[i] < pivot)\
			arr[m++] = arr[i++];/*m에 데이터를 복사*/\
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

int nrData{ 0 }; //현재 총 데이터 개수
int originNrData{ 0 }; //원본 총 데이터 개수

__m256i shuffle_mask[256];
__m256i rshuffle_mask[256];
int songShuffle_mask[256] = { 0, };
clock_t begin, end;

/* INFO: 랜덤값 생성기 클래스 */
class random_generator {
private:
	//INFO: 시드값을 얻기 위한 변수
	std::random_device rd;
	//INFO: 난수생성 엔진 초기화
	mt19937 gen;
	//INFO: 균등하게 나타나는 난수열을 생성하기 위해 균등 분포 정의
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

/* INFO: SIMD에서 활용할 룩업테이블 생성 */
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
	//rshuffle_mask는 피봇과 같거나 큰 데이터들을 원본 배열에 저장할 때 데이터들을 오른쪽 끝에 붙이기 위해 테이블을 따로 만들어둠(예: -1 -1 -1 1 0)
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

/* INFO: 정해진 배열에 대해 값을 랜덤하게 초기화 */
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

/* INFO: 스칼라 방식의 퀵정렬 */
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

			stack[top++] = high; //새로운 범위 low와 high를 입력
			stack[top++] = i + 1;
			stack[top++] = i - 1;
			stack[top++] = low;
		}
	}
}

/* INFO: SIMD 256bit 기반의 퀵정렬 */
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

	int mc = 0;//피봇과 같은 값의 개수
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

	while (top != 0) { //만약 스택에 데이터가 있다면
		low = (top != 0) ? stack[--top] : 0;
		high = (top != 0) ? stack[--top] : 0;
		if (high > low) {//high가 low보다 크다면 
			mc = high - low; //low를 제외한 갯수
			b = (low + high) >> 1;//middle element
			m = low + 1;//피봇보다 작은 데이터를 저장할 현재 위치
			n = high + 1;//피봇과 같거나 큰 데이터를 저장할 현재 위치

			if (SDTRP > mc) {

				b = MED3(low, b, high);
				SWAP(low, b);
				pivot = arr[low];
				cnt = (mc >> NUM_SIMD_SHIFT);
				if (cnt == 0) {
					SMALLLOC;
				}
				else {
					simdPivot = _mm256_set1_epi32(pivot);//피봇 값을 SIMD레지스터에 중복적재
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
				simdPivot = _mm256_set1_epi32(pivot);//피봇 값을 SIMD레지스터에 중복적재

				simdBuffer[0] = _mm256_loadu_si256((__m256i*) & arr[lrPick[0]]); //비교할 데이터들을 차례대로 적재
				lrPick[0] += NUM_SIMD_DATA;
				simdBuffer[1] = _mm256_loadu_si256((__m256i*) & arr[lrPick[1]]); //비교할 데이터들을 차례대로 적재
				lrPick[1] -= NUM_SIMD_DATA;
				simdBuffer[2] = _mm256_loadu_si256((__m256i*) & arr[lrPick[1]]); //비교할 데이터들을 차례대로 적재
				lrPick[1] -= NUM_SIMD_DATA;
				bfIdx = 0;

				while (1) {
					SIMDOP;
					if (lrPick[0] <= lrPick[1]) {
						cmp = (lrPick[0] - m) >= NUM_SIMD_DATA; //좌측 공간이 없으면 채우고 남으면 오른쪽을 채움. 양쪽다 부족할 수는 없음
						simdBuffer[bfIdx] = _mm256_loadu_si256((__m256i*) & arr[lrPick[cmp]]); //비교할 데이터들을 차례대로 적재
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

			newPivot = m - 1;//피봇 위치 설정
			SWAP(newPivot, low);
			for (int i = m; i < n; i++) arr[i] = arr[newPivot];

			stack[top++] = high; //새로운 범위 low와 high를 입력
			stack[top++] = n;
			stack[top++] = newPivot - 1;
			stack[top++] = low;
		}
	}
}

/* INFO: 주어진 배열에 대해 작은값, 같은값, 큰값으로 분류 */
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

	int mc = 0;//피봇과 같은 값의 개수
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

	while (top != 0) { //만약 스택에 데이터가 있다면
		low = (top != 0) ? stack[--top] : 0;
		high = (top != 0) ? stack[--top] : 0;
		pvtLow = (pvtTop != 0) ? pvtStack[--pvtTop] : 0;
		pvtHigh = (pvtTop != 0) ? pvtStack[--pvtTop] : 0;
		if ((high > low) && (pvtHigh >= pvtLow)) {//high가 low보다 크다면 
			pvtIdx = (pvtHigh + pvtLow) >> 1;
			pivot = rngBndry[pvtIdx]; //2로 나눔

			mc = high - low + 1; //low를 제외한 갯수
			m = low;//피봇보다 작은 데이터를 저장할 현재 위치
			n = high + 1;//피봇과 같거나 큰 데이터를 저장할 현재 위치
			if (SDTRP > mc) {
				cnt = (mc >> NUM_SIMD_SHIFT);
				if (cnt == 0) {
					SMALLLOC;
				}
				else {
					simdPivot = _mm256_set1_epi32(pivot);//피봇 값을 SIMD레지스터에 중복적재
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
				simdPivot = _mm256_set1_epi32(pivot);//피봇 값을 SIMD레지스터에 중복적재
				simdBuffer[0] = _mm256_loadu_si256((__m256i*) & arr[lrPick[0]]); //비교할 데이터들을 차례대로 적재
				lrPick[0] += NUM_SIMD_DATA;
				simdBuffer[1] = _mm256_loadu_si256((__m256i*) & arr[lrPick[1]]); //비교할 데이터들을 차례대로 적재
				lrPick[1] -= NUM_SIMD_DATA;
				simdBuffer[2] = _mm256_loadu_si256((__m256i*) & arr[lrPick[1]]); //비교할 데이터들을 차례대로 적재
				lrPick[1] -= NUM_SIMD_DATA;
				bfIdx = 0;

				while (1) {
					SIMDOP;
					if (lrPick[0] <= lrPick[1]) {
						cmp = (lrPick[0] - m) >= NUM_SIMD_DATA; //좌측 공간이 없으면 채우고 남으면 오른쪽을 채움. 양쪽다 부족할 수는 없음
						simdBuffer[bfIdx] = _mm256_loadu_si256((__m256i*) & arr[lrPick[cmp]]); //비교할 데이터들을 차례대로 적재
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

			stack[top++] = high; //새로운 범위 low와 high를 입력
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

/* INFO: 1차적으로 KDE를 통해 정해진 범위에 대해 데이터를 분류함 */
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

/* INFO: 각 스레드가 주어진 페이로드에 대해 정렬 수행 */
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

/* INFO: KDE 기법을 이용하여 값의 분포도를 계산 */
void calculateKDE(int* arr, double sectionDensity[], int nrData) {
	double sampleArr[NUM_SAMPLES];	//샘플링 데이터 arr
	double stdDeviation = 0.0;//표준편차
	double avg = 0.0;//평균
	double bandwidth = 0.0;//대역폭
	int randomPos;

	int scaleFactor;
	if (nrData > RDMAX) scaleFactor = (nrData - RDMAX) / RDMAX;
	else scaleFactor = 0;
	for (int i = 0; i < NUM_SAMPLES; i++) { //샘플 값을 보관하는 변수 sampleArr에 임의의 arr 값을 저장
		randomPos = rand() * scaleFactor + rand() % nrData;
		sampleArr[i] = (double)(arr[randomPos]);
		avg += sampleArr[i];
	}
	avg /= NUM_SAMPLES;//평균값
	double temp;
	for (int i = 0; i < NUM_SAMPLES; i++) {
		temp = sampleArr[i] - avg;
		stdDeviation += (temp * temp);
	}
	stdDeviation /= NUM_SAMPLES;
	stdDeviation = sqrt(stdDeviation);//표준편차
	bandwidth = 1.06 * stdDeviation * pow(NUM_SAMPLES, -0.2);//대역폭은 위키피디아를 참조하여 적용함. 나중에 실험할 때 변경가능
	int intervalOfSection = maxValue / NUM_SECTIONS;
	double hSMPgop = (NUM_SAMPLES * bandwidth);
	temp = 0.0;

	for (int i = 0; i < NUM_SECTIONS; i++) {//구간수
		for (int j = 0; j < NUM_SAMPLES; j++) { //샘플수
			temp = ((double)(i * intervalOfSection) - sampleArr[j]) / bandwidth; //여기까지 u 값
			temp *= -0.5 * temp;
			sectionDensity[i] += 399000000.0 * pow(2.7183, temp); //값이 너무 작게 나와서 배수를 늘림
		}
		sectionDensity[i] /= hSMPgop;
	}
}

/* INFO: 단일 배열에 대한 균등부하분배 계산 */
void distributeWorkloadWithSingleArray(int* arr, int* rngBndry, int nrData) {
	//INFO: 배열에서 KDE를 통해 추정된 각 구간의 밀도
	double* sectionDensity = (double*)malloc(sizeof(double) * NUM_SECTIONS);
	//INFO: 밀도 총합
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

/* INFO: 실제 정렬을 시작하는 함수 */
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
	printf("멀티스레드 작업 분배 계산 시간 : %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);
	totalTime += (double)(end - begin);

	begin = clock();
	SIPQ(&vArr, nrData, rngPosArrR, rngBndry);
	end = clock();
	printf("멀티스레드 정렬 수행 시간 : %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);
	totalTime += (double)(end - begin);

	//INFO: 일부 데이터만 표출 (확인용)
	int valInterval = nrData > 10000 ? nrData / 1000 : 1;
	for (int i = 0; i < nrData; i+= valInterval) {
		printf("%d ", vArr[i]);
	}
	printf("\n");

	printf("firstValue : %d, lastValue : %d\n", vArr[0], vArr[originNrData - 1]);

	//INFO : SIMD를 위해 추가적으로 할당된 공간을 버리고 원본 크기로 전환
	nrData = originNrData;

	printf("Finish quicksort\n");

	printf("전체 수행 시간 : %lf\n", (double)(totalTime / CLOCKS_PER_SEC));

	free(rngPosArrR);
	free(vArr);
	free(rngBndry);
}