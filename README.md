# SIPQ
 - SIMD-based In-Place Quicksort
 - SIPQ는 SIMD 기반의 제자리 퀵정렬을 수행합니다. 관련 논문은 [여기](https://lib.koreatech.ac.kr/#/search/detail/742017)에서 확인하실 수 있습니다.
 - 현재 AVX2 명령어셋에서 제공하는 256 비트 크기의 SIMD 레지스터를 활용하여 퀵정렬을 수행합니다.
 - 멀티스레드를 지원합니다.
