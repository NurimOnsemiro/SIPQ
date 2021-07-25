# SIPQ
 - SIMD-based In-Place Quicksort
 - SIPQ는 SIMD 기반의 제자리 퀵정렬을 수행합니다. 관련 논문은 [여기](https://lib.koreatech.ac.kr/#/search/detail/742017)에서 확인하실 수 있습니다.
 - 현재 AVX2 명령어셋에서 제공하는 256 비트 크기의 SIMD 레지스터를 활용하여 퀵정렬을 수행합니다. (CPU에서 AVX2 명령어셋이 제공되지 않으면 사용할 수 없습니다.)
 - 멀티스레드를 지원합니다.

#작업순서
1. 주어진 배열에 대해 특정 개수만큼 임의의 위치의 값을 샘플링한 후 KDE를 계산하여 값의 분포도를 추정한다.
2. 값의 분포도를 바탕으로 스레드 개수 만큼 부하를 균등하게 분배한다.
3. 배열을 특정 개수의 파티션으로 분할한 후 각 스레드는 모든 파티션들에 대해 KDE를 통해 계산된 값의 범위들에 대해 값을 1차적으로 분류한다.
4. 1차 분류가 완료된 각 파티션을 순회하면서 동일한 범위 내의 데이터들을 복사하여 특정 스레드에게 할당한다.
5. 이후 각 스레드는 주어진 값의 범위에 대해 병렬적으로 SIMD 기반의 퀵정렬을 수행한다.
6. 각 스레드에게 주어진 범위에 대한 정렬이 완료되면 최종적으로 하나의 정렬된 배열을 획득한다.

#결과
 - 스칼라 퀵정렬보다 SIMD 퀵정렬이 최소 2배 이상 빠른 성능을 보인다.

#추후계획
 - AVX-512를 이용하여 더 빠른 SIMD 기반의 퀵정렬을 구현해볼 계획입니다.
