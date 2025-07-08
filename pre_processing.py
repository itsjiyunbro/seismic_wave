## Conventional Preprocessing

# 1. ObsPy 설치
!pip install obspy

# 2. 수동으로 런타임 재시작
# 메뉴: Runtime → Restart runtime

# 3. 재시작 후 import 테스트
import obspy
print(f"✅ ObsPy 버전: {obspy.__version__}")



import numpy as np
import obspy
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

print("=== 크래시 안전 전통적인 지진파 전처리 6단계 ===")
print("Demultiplexing → Trace Editing → Gain Recovery → Filtering → Deconvolution → CMP Gather")

# 데이터 로딩
print("\n원시 데이터 로딩...")
stream = obspy.read("ANMO_sample.mseed")
print(f"✅ 로딩 완료: {len(stream)}개 트레이스")

# 원본 데이터 백업
original_stream = stream.copy()



print("\n=== 3채널 데이터 시각화 준비 ===")

# 시간 축 생성
time_axis = {}
for i, trace in enumerate(stream):
    sampling_rate = trace.stats.sampling_rate
    num_samples = len(trace.data)
    duration = num_samples / sampling_rate
    time_axis[trace.stats.channel] = np.linspace(0, duration, num_samples)

    print(f"{trace.stats.channel} 채널:")
    print(f"  시간 범위: 0 ~ {duration:.1f}초")
    print(f"  데이터 범위: {trace.data.min():.1f} ~ {trace.data.max():.1f}")

print("\n시각화 코드 (matplotlib 사용 시):")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 8))
for i, trace in enumerate(stream):
    channel = trace.stats.channel
    time = time_axis[channel]
    axes[i].plot(time, trace.data)
    axes[i].set_title(f'{channel} 채널')
    axes[i].set_ylabel('진폭')
    if i == 2:
        axes[i].set_xlabel('시간 (초)')
plt.tight_layout()
plt.show()



# =====================================================
# 1단계: Demultiplexing (역다중화)
# 이미 데이터가 3개의 트레이스에서 BH1, BH2, BHZ로 분리되어있기 때문에 확인하고 정리하는 단계로 여기면 된다
# =====================================================
print("\n🔸 1단계: Demultiplexing (역다중화)")
print("- 다채널 지진파 데이터를 개별 채널로 분리")

demux_channels = {}
for i, trace in enumerate(stream):
    channel_id = trace.stats.channel
    demux_channels[channel_id] = {
        'trace': trace,
        'network': trace.stats.network,
        'station': trace.stats.station,
        'channel': trace.stats.channel,
        'sampling_rate': trace.stats.sampling_rate,
        'npts': trace.stats.npts,
        'starttime': trace.stats.starttime
    }
    print(f"  📊 {channel_id}: {trace.stats.sampling_rate}Hz, {trace.stats.npts} samples")

print(f"✅ 1단계 완료: {len(demux_channels)}개 채널 분리")



# =====================================================
# 2단계: Trace Editing (트레이스 편집)
# =====================================================
print("\n🔸 2단계: Trace Editing (트레이스 편집)")
print("- 불량 데이터 제거 및 품질 관리")

edited_channels = {}
for channel_id, ch_data in demux_channels.items():
    trace = ch_data['trace']
    data = trace.data.copy()

    print(f"  🔍 {channel_id} 품질 검사...")

    # Dead trace 검사
    if np.all(data == 0) or np.var(data) < 1e-12:
        print(f"    ❌ Dead trace - 제거")
        continue

    # NaN/Inf 값 검사
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        print(f"    ⚠️ NaN/Inf 값 발견 - 보정")
        # NaN을 0으로, Inf를 클리핑
        data = np.nan_to_num(data, nan=0.0, posinf=np.max(data[np.isfinite(data)]),
                            neginf=np.min(data[np.isfinite(data)]))

    # 스파이크 제거 (Z-score > 5)
    z_scores = np.abs((data - np.mean(data)) / (np.std(data) + 1e-10))
    spike_count = np.sum(z_scores > 5)
    if spike_count > 0:
        print(f"    ⚠️ {spike_count}개 스파이크 제거")
        spike_mask = z_scores > 5
        # 스파이크를 주변 값의 평균으로 대체
        for idx in np.where(spike_mask)[0]:
            if idx > 0 and idx < len(data) - 1:
                data[idx] = (data[idx-1] + data[idx+1]) / 2

    # 편집된 데이터 저장
    edited_trace = trace.copy()
    edited_trace.data = data
    edited_channels[channel_id] = edited_trace
    print(f"    ✅ 품질 검사 통과")

print(f"✅ 2단계 완료: {len(edited_channels)}개 채널 유지")



print("=== 포화 보간이 실패한 이유 분석 ===")

# 포화 보간 로직 분석
for channel_id, ch_data in demux_channels.items():
    data = ch_data['trace'].data

    print(f"\n🔍 {channel_id} 포화 분석:")

    max_val = np.max(np.abs(data))
    saturation_threshold = max_val * 0.95
    saturated_mask = np.abs(data) >= saturation_threshold
    saturated_indices = np.where(saturated_mask)[0]

    print(f"    최대값: {max_val:.2f}")
    print(f"    포화 임계값: {saturation_threshold:.2f}")
    print(f"    포화 값 개수: {len(saturated_indices)}")

    if len(saturated_indices) > 0:
        print(f"    포화 값 위치: {saturated_indices[:10]}...")  # 처음 10개만

        # 보간 실패 원인 분석
        edge_count = 0
        cluster_count = 0
        successful_count = 0

        for idx in saturated_indices:
            if idx <= 5 or idx >= len(data) - 5:
                edge_count += 1
            else:
                # 주변에도 포화 값이 있는지 확인
                before_vals = data[idx-5:idx][~saturated_mask[idx-5:idx]]
                after_vals = data[idx+1:idx+6][~saturated_mask[idx+1:idx+6]]

                if len(before_vals) == 0 or len(after_vals) == 0:
                    cluster_count += 1
                else:
                    successful_count += 1

        print(f"    보간 실패 원인:")
        print(f"      경계 근처: {edge_count}개")
        print(f"      포화 클러스터: {cluster_count}개")
        print(f"      보간 가능: {successful_count}개")

        # 실제 보간 가능한 것만 처리
        if successful_count > 0:
            print(f"    💡 실제로는 {successful_count}개 보간 가능했음!")



print("\n=== 개선된 포화 처리 ===")

# 더 똑똑한 포화 처리
really_improved_edited = {}

for channel_id, ch_data in demux_channels.items():
    trace = ch_data['trace']
    data = trace.data.copy()

    print(f"\n🛠️ {channel_id} 개선된 편집:")

    total_edits = 0

    # 1. 더 관대한 포화 처리
    max_val = np.max(np.abs(data))
    saturation_threshold = max_val * 0.98  # 더 엄격한 기준
    saturated_indices = np.where(np.abs(data) >= saturation_threshold)[0]

    if len(saturated_indices) > 0:
        print(f"    🔴 {len(saturated_indices)}개 포화 값 발견")

        saturation_edits = 0
        for idx in saturated_indices:
            # 더 작은 윈도우로 보간 시도
            if idx > 2 and idx < len(data) - 2:
                # 포화되지 않은 가장 가까운 값들 찾기
                left_val = None
                right_val = None

                # 왼쪽 탐색
                for i in range(idx-1, max(0, idx-10), -1):
                    if np.abs(data[i]) < saturation_threshold:
                        left_val = data[i]
                        break

                # 오른쪽 탐색
                for i in range(idx+1, min(len(data), idx+10)):
                    if np.abs(data[i]) < saturation_threshold:
                        right_val = data[i]
                        break

                # 보간 수행
                if left_val is not None and right_val is not None:
                    data[idx] = (left_val + right_val) / 2
                    saturation_edits += 1
                elif left_val is not None:
                    data[idx] = left_val * 0.9  # 약간 감소
                    saturation_edits += 1
                elif right_val is not None:
                    data[idx] = right_val * 0.9  # 약간 감소
                    saturation_edits += 1

        print(f"    ✅ {saturation_edits}개 포화 값 보간")
        total_edits += saturation_edits

    # 2. 더 효과적인 드리프트 제거
    window_size = 400  # 10초 윈도우로 줄임
    if len(data) > window_size:
        # 다항식 detrend (2차)
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 2)  # 2차 다항식
        trend = np.polyval(coeffs, x)
        detrended_data = data - trend

        # 드리프트 제거 효과
        original_drift = np.std([np.mean(data[i:i+400]) for i in range(0, len(data)-400, 400)])
        new_drift = np.std([np.mean(detrended_data[i:i+400]) for i in range(0, len(detrended_data)-400, 400)])

        print(f"    📉 2차 다항식 detrend: {original_drift:.2f} → {new_drift:.2f}")
        data = detrended_data

    # 3. 스파이크 제거 (2.5σ 기준으로 더 엄격하게)
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    spike_mask = z_scores > 2.5
    spike_count = np.sum(spike_mask)

    if spike_count > 0:
        print(f"    ⚡ {spike_count}개 스파이크 제거 (2.5σ 기준)")
        for idx in np.where(spike_mask)[0]:
            if idx > 1 and idx < len(data) - 1:
                # 중앙값 필터 적용
                data[idx] = np.median(data[idx-1:idx+2])
        total_edits += spike_count

    # 4. 추가: 고주파 노이즈 감소
    # 간단한 3점 이동평균으로 고주파 노이즈 감소
    smoothed_data = np.convolve(data, [0.25, 0.5, 0.25], mode='same')
    noise_reduction = np.std(data - smoothed_data)

    if noise_reduction > np.std(data) * 0.1:  # 10% 이상의 노이즈
        print(f"    🔇 고주파 노이즈 감소: {noise_reduction:.2f}")
        data = smoothed_data

    # 결과 저장
    edited_trace = trace.copy()
    edited_trace.data = data
    really_improved_edited[channel_id] = edited_trace

    print(f"    ✅ 총 편집: {total_edits}개 값 수정")

print(f"\n🎯 진짜 개선된 편집 결과!")



# =====================================================
# 3단계: Gain Recovery (이득 복구)
# =====================================================
print("\n🔸 3단계: Gain Recovery (이득 복구)")
print("- 기록 시 적용된 이득을 보상하여 원래 진폭 복원")

gain_recovered = {}
for channel_id, trace in edited_channels.items():
    data = trace.data.copy()

    print(f"  ⚡ {channel_id} 이득 복구...")

    # 1. 기하학적 확산 보정 (r^-1 법칙)
    # 일반적인 거리 100km 가정
    distance_km = 100
    geometric_factor = distance_km  # 거리비례 보정

    # 2. 매체 감쇠 보정 (Q factor 보정)
    # 주파수 영역에서 감쇠 보정
    Q_factor = 200  # 일반적인 Q값
    velocity_km_s = 3.5  # S파 속도

    # FFT로 주파수 영역 변환
    dt = 1.0 / trace.stats.sampling_rate
    freqs = np.fft.fftfreq(len(data), dt)
    fft_data = np.fft.fft(data)

    # 감쇠 보정 적용 (0 주파수 제외)
    attenuation_correction = np.ones_like(freqs)
    non_zero_freq = freqs != 0
    attenuation_correction[non_zero_freq] = np.exp(
        np.pi * np.abs(freqs[non_zero_freq]) * distance_km / (Q_factor * velocity_km_s)
    )

    # 보정 적용 및 역변환
    corrected_fft = fft_data * attenuation_correction
    corrected_data = np.real(np.fft.ifft(corrected_fft))

    # 3. 기하학적 확산 보정 적용
    final_data = corrected_data * geometric_factor

    # 4. 계기 응답 제거 (간단한 고역통과)
    # 매우 낮은 주파수 성분 제거 (0.01Hz 이하)
    if trace.stats.sampling_rate > 0.02:  # 나이퀴스트 조건
        # 수동으로 고역통과 필터 구현 (scipy 사용)
        nyquist = trace.stats.sampling_rate / 2
        low_cutoff = min(0.01, nyquist * 0.01)  # 안전한 cutoff

        try:
            b, a = signal.butter(2, low_cutoff / nyquist, btype='high')
            final_data = signal.filtfilt(b, a, final_data)
        except:
            print(f"    ⚠️ 고역통과 필터 실패 - 건너뜀")

    # 결과 저장
    recovered_trace = trace.copy()
    recovered_trace.data = final_data
    gain_recovered[channel_id] = recovered_trace

    print(f"    ✅ 이득 복구 완료 (증폭: {geometric_factor:.1f}x)")

print(f"✅ 3단계 완료: {len(gain_recovered)}개 채널 이득 복구")



# =====================================================
# 4단계: Filtering (필터링)
# =====================================================
print("\n🔸 4단계: Filtering (필터링)")
print("- 주파수 영역에서 노이즈 제거")

filtered_channels = {}
for channel_id, trace in gain_recovered.items():
    data = trace.data.copy()

    print(f"  🎛️ {channel_id} 필터링...")

    # 1. 선형 트렌드 제거 (수동 구현)
    x = np.arange(len(data))
    if len(data) > 1:
        slope = np.sum((x - np.mean(x)) * (data - np.mean(data))) / np.sum((x - np.mean(x))**2)
        intercept = np.mean(data) - slope * np.mean(x)
        trend = slope * x + intercept
        detrended_data = data - trend
    else:
        detrended_data = data

    # 2. 밴드패스 필터 (1-20Hz) - scipy 직접 사용
    sampling_rate = trace.stats.sampling_rate
    nyquist = sampling_rate / 2

    # 안전한 주파수 범위 설정
    low_freq = min(1.0, nyquist * 0.1)
    high_freq = min(20.0, nyquist * 0.9)

    if low_freq < high_freq and nyquist > low_freq:
        try:
            # 버터워스 밴드패스 필터
            b, a = signal.butter(4, [low_freq/nyquist, high_freq/nyquist], btype='band')
            bandpass_data = signal.filtfilt(b, a, detrended_data)
            print(f"    ✅ 밴드패스 필터 적용: {low_freq:.1f}-{high_freq:.1f}Hz")
        except Exception as e:
            print(f"    ⚠️ 밴드패스 필터 실패: {e}")
            bandpass_data = detrended_data
    else:
        print(f"    ⚠️ 부적절한 주파수 범위 - 필터 건너뜀")
        bandpass_data = detrended_data

    # 3. 노치 필터 (60Hz 전력선 간섭 제거)
    if sampling_rate > 120:  # 나이퀴스트 조건
        try:
            notch_freq = 60.0
            Q = 30
            b, a = signal.iirnotch(notch_freq, Q, sampling_rate)
            notched_data = signal.filtfilt(b, a, bandpass_data)
            print(f"    ✅ 노치 필터 적용: {notch_freq}Hz")
        except Exception as e:
            print(f"    ⚠️ 노치 필터 실패: {e}")
            notched_data = bandpass_data
    else:
        notched_data = bandpass_data

    # 결과 저장
    filtered_trace = trace.copy()
    filtered_trace.data = notched_data
    filtered_channels[channel_id] = filtered_trace

print(f"✅ 4단계 완료: {len(filtered_channels)}개 채널 필터링")



# =====================================================
# 5단계: Deconvolution (역컨볼루션)
# =====================================================
print("\n🔸 5단계: Deconvolution (역컨볼루션)")
print("- 지진파 전파 과정에서 발생한 파형 왜곡 보정")

deconvolved_channels = {}
for channel_id, trace in filtered_channels.items():
    data = trace.data.copy()

    print(f"  🔄 {channel_id} 역컨볼루션...")

    # 1. 예측 역컨볼루션 (간단한 형태)
    # 자기상관함수 기반 예측 필터
    autocorr_length = min(100, len(data) // 10)  # 안전한 길이

    if len(data) > autocorr_length * 2:
        # 자기상관함수 계산
        autocorr = np.correlate(data, data, mode='full')
        center = len(autocorr) // 2
        autocorr = autocorr[center:center + autocorr_length]

        # 예측 필터 설계 (간단한 형태)
        if len(autocorr) > 1 and np.std(autocorr) > 0:
            # 정규화된 자기상관
            autocorr = autocorr / autocorr[0]

            # 간단한 예측 필터 (2차)
            if len(autocorr) >= 3:
                pred_filter = np.array([1, -autocorr[1], autocorr[2]/2])
            else:
                pred_filter = np.array([1, -0.5])
        else:
            pred_filter = np.array([1])

        # 예측 역컨볼루션 적용
        try:
            deconv_data = signal.lfilter(pred_filter, [1], data)
        except:
            deconv_data = data
    else:
        deconv_data = data

    # 2. 스파이킹 역컨볼루션 효과 (고주파 성분 강화)
    # 1차 미분으로 고주파 강화
    diff_data = np.diff(deconv_data, prepend=deconv_data[0])

    # 원본과 미분의 가중합 (70% 원본 + 30% 고주파 강화)
    enhanced_data = 0.7 * deconv_data + 0.3 * diff_data

    # 결과 저장
    deconv_trace = trace.copy()
    deconv_trace.data = enhanced_data
    deconvolved_channels[channel_id] = deconv_trace

    print(f"    ✅ 역컨볼루션 완료")

print(f"✅ 5단계 완료: {len(deconvolved_channels)}개 채널 역컨볼루션")



# =====================================================
# 6단계: CMP Gather (공통 중점 집합)
# =====================================================
print("\n🔸 6단계: CMP Gather (공통 중점 집합)")
print("- 같은 지하 점을 반사한 신호들을 그룹화")

# 채널별로 그룹화 (Z, N, E 성분)
gathered_channels = {}
channel_groups = {
    'vertical': [],    # Z 성분
    'horizontal_1': [], # N, 1 성분
    'horizontal_2': []  # E, 2 성분
}

for channel_id, trace in deconvolved_channels.items():
    if 'Z' in channel_id:
        channel_groups['vertical'].append((channel_id, trace))
    elif 'N' in channel_id or '1' in channel_id:
        channel_groups['horizontal_1'].append((channel_id, trace))
    elif 'E' in channel_id or '2' in channel_id:
        channel_groups['horizontal_2'].append((channel_id, trace))

# 각 그룹별로 대표 채널 선택 (또는 스택)
for group_name, traces in channel_groups.items():
    if traces:
        if len(traces) == 1:
            # 단일 채널
            channel_id, trace = traces[0]
            gathered_channels[f"{group_name}_{channel_id}"] = trace
            print(f"  📍 {group_name}: {channel_id} 선택")
        else:
            # 여러 채널이 있으면 첫 번째만 선택 (또는 스택 가능)
            channel_id, trace = traces[0]
            gathered_channels[f"{group_name}_{channel_id}"] = trace
            print(f"  📍 {group_name}: {channel_id} 선택 ({len(traces)}개 중)")

print(f"✅ 6단계 완료: {len(gathered_channels)}개 그룹 형성")



# =====================================================
# 최종 결과
# =====================================================
print("\n🎉 전통적인 지진파 전처리 6단계 완료!")
print("="*50)
print("처리 결과:")
for step, count in [
    ("1. Demultiplexing", len(demux_channels)),
    ("2. Trace Editing", len(edited_channels)),
    ("3. Gain Recovery", len(gain_recovered)),
    ("4. Filtering", len(filtered_channels)),
    ("5. Deconvolution", len(deconvolved_channels)),
    ("6. CMP Gather", len(gathered_channels))
]:
    print(f"  {step}: {count}개 채널")

print("\n최종 출력:")
final_processed_data = {}
for channel_name, trace in gathered_channels.items():
    final_processed_data[channel_name] = {
        'data': trace.data,
        'sampling_rate': trace.stats.sampling_rate,
        'channel': trace.stats.channel,
        'length': len(trace.data)
    }
    print(f"  🔸 {channel_name}: {len(trace.data)} samples @ {trace.stats.sampling_rate}Hz")

print("\n✅ 전통적인 전처리 완료 - 딥러닝 전처리 단계로 진행 가능!")



## 2. Preprocessing for Deep Learning

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("🚀 딥러닝용 지진파 전처리 시작!")
print("="*60)

# ============================================================================
# 1단계: 데이터 준비 및 검증
# ============================================================================
print("\n📋 1단계: 데이터 준비 및 검증")

# 전처리된 지진파 데이터 확인
print("전처리된 지진파 데이터:")
for channel_name, data_info in final_processed_data.items():
    print(f"  🔸 {channel_name}: {data_info['length']} samples @ {data_info['sampling_rate']}Hz")

# 카탈로그 데이터 확인
if 'catalog' in locals() and catalog is not None:
    print(f"\n카탈로그 데이터:")
    print(f"  📊 {len(catalog)}개 지진 이벤트")
    print(f"  📅 컬럼: {list(catalog.columns)}")
    if 'magnitude' in catalog.columns:
        print(f"  📈 규모 범위: {catalog['magnitude'].min()} - {catalog['magnitude'].max()}")
else:
    print("⚠️ 카탈로그 데이터 로딩 필요")

    # 실제 정답데이터 파일 읽기
    try:
        import pandas as pd

        # Excel 파일 읽기 (헤더는 2번째 행, 데이터는 4번째 행부터)
        raw_catalog = pd.read_excel("inCountryEarthquakeList_20060101_20250704.xlsx",
                                   header=1, skiprows=[2])  # 2번째 행을 헤더로, 3번째 행 건너뛰기

        # 데이터 정리
        # 유효한 데이터만 필터링 (number와 origin_time이 있는 행)
        valid_mask = raw_catalog['number'].notna() & raw_catalog['origin_time'].notna()
        catalog_clean = raw_catalog[valid_mask].copy()

        # 위도/경도 문자열 처리 ("36.85 N" -> 36.85)
        if 'latitude' in catalog_clean.columns:
            catalog_clean['latitude'] = catalog_clean['latitude'].astype(str).str.replace(' N', '').str.replace(' S', '').astype(float)
        if 'longitude' in catalog_clean.columns:
            catalog_clean['longitude'] = catalog_clean['longitude'].astype(str).str.replace(' E', '').str.replace(' W', '').astype(float)

        # origin_time을 datetime으로 변환 (Excel 숫자를 날짜로)
        # Excel의 숫자 날짜를 pandas datetime으로 변환
        catalog_clean['origin_time'] = pd.to_datetime(catalog_clean['origin_time'], origin='1899-12-30', unit='D')

        # 최종 카탈로그
        catalog = catalog_clean.reset_index(drop=True)

        print(f"✅ 정답데이터 로딩 완료!")
        print(f"  📊 총 {len(catalog)}개 지진 이벤트")
        print(f"  📅 컬럼: {list(catalog.columns)}")
        print(f"  📈 규모 범위: {catalog['magnitude'].min():.1f} - {catalog['magnitude'].max():.1f}")
        print(f"  📍 위치 범위:")
        print(f"    위도: {catalog['latitude'].min():.2f}°N - {catalog['latitude'].max():.2f}°N")
        print(f"    경도: {catalog['longitude'].min():.2f}°E - {catalog['longitude'].max():.2f}°E")
        print(f"  📅 시간 범위: {catalog['origin_time'].min()} ~ {catalog['origin_time'].max()}")

        print(f"\n첫 3개 이벤트:")
        print(catalog[['number', 'origin_time', 'magnitude', 'depth', 'latitude', 'longitude', 'location']].head(3))

    except Exception as e:
        print(f"❌ 정답데이터 로딩 실패: {str(e)}")
        # 기본 테스트 데이터 생성
        catalog = pd.DataFrame({
            'number': [1, 2],
            'origin_time': pd.to_datetime(['2022-01-01T10:00:00', '2022-01-02T15:30:00']),
            'magnitude': [2.7, 3.5],
            'depth': [10.0, 8.0],
            'latitude': [36.123, 36.789],
            'longitude': [127.456, 128.123],
            'location': ['테스트 지역 1', '테스트 지역 2']
        })
        print(f"  📊 테스트 데이터 생성: {len(catalog)}개 이벤트")

# ============================================================================
# 2단계: 3채널 데이터 결합
# ============================================================================
print(f"\n🔗 2단계: 3채널 데이터 결합")

# 채널 순서 정의 (Z, 1, 2 순서로)
channel_order = ['vertical_BHZ', 'horizontal_1_BH1', 'horizontal_2_BH2']
combined_channels = []

print("채널 결합 진행:")
for channel_name in channel_order:
    if channel_name in final_processed_data:
        data = final_processed_data[channel_name]['data']
        combined_channels.append(data)
        print(f"  ✅ {channel_name}: {len(data)} samples 추가")
    else:
        print(f"  ❌ {channel_name}: 채널 없음")

if len(combined_channels) == 3:
    # (시간, 채널) 형태로 결합
    combined_data = np.column_stack(combined_channels)
    print(f"✅ 3채널 결합 완료: {combined_data.shape} (시간 x 채널)")

    # 기본 통계
    print(f"  📊 데이터 범위: {combined_data.min():.3f} ~ {combined_data.max():.3f}")
    print(f"  📊 평균: {np.mean(combined_data):.3f}")
    print(f"  📊 표준편차: {np.std(combined_data):.3f}")
else:
    print("❌ 3채널 결합 실패")
    combined_data = None

# ============================================================================
# 3단계: 시간 기준 윈도잉
# ============================================================================
print(f"\n⏰ 3단계: 시간 기준 윈도잉")

# 윈도우 파라미터 설정
sampling_rate = 40  # Hz
window_duration = 20  # 초
window_samples = int(window_duration * sampling_rate)  # 800 samples
overlap_ratio = 0.5  # 50% 겹침
overlap_samples = int(window_samples * overlap_ratio)

print(f"윈도우 설정:")
print(f"  🕐 윈도우 길이: {window_duration}초 ({window_samples} samples)")
print(f"  🔄 겹침: {overlap_ratio*100}% ({overlap_samples} samples)")

# 카탈로그 기반 윈도잉 함수
def create_earthquake_windows(combined_data, catalog, sampling_rate,
                             window_samples, before_seconds=10, after_seconds=10):
    """지진 이벤트 기준으로 윈도우 생성"""

    windows = []
    labels = []
    metadata = []

    print(f"\n지진 이벤트별 윈도우 생성:")

    # pandas DataFrame을 안전하게 순회
    for idx in range(len(catalog)):
        try:
            # iloc을 사용하여 안전하게 행 접근
            event = catalog.iloc[idx]
            magnitude = float(event['magnitude'])

            print(f"  📍 이벤트 {idx+1}: M{magnitude}")

            # 시간 정보 (실제로는 카탈로그의 시간과 지진파 데이터의 시간을 매칭해야 함)
            # 여기서는 예시로 데이터 중앙 부분을 지진 발생 시점으로 가정
            total_samples = len(combined_data)
            earthquake_sample = total_samples // 2  # 중앙점을 지진 발생으로 가정

            # 지진 전후 구간 계산
            before_samples = int(before_seconds * sampling_rate)
            after_samples = int(after_seconds * sampling_rate)

            start_sample = earthquake_sample - before_samples
            end_sample = earthquake_sample + after_samples

            # 유효 범위 확인
            if start_sample >= 0 and end_sample <= total_samples:
                event_window = combined_data[start_sample:end_sample]

                # 윈도우가 충분히 긴지 확인
                if len(event_window) >= window_samples:
                    # 이벤트 윈도우 내에서 슬라이딩 윈도우 생성
                    event_window_count = 0
                    overlap_samples = window_samples // 2  # 50% 겹침

                    for i in range(0, len(event_window) - window_samples + 1, overlap_samples):
                        window = event_window[i:i + window_samples]

                        if len(window) == window_samples:
                            windows.append(window)

                            # 안전하게 라벨 생성
                            label_dict = {
                                'magnitude': magnitude,
                                'depth': float(event['depth']),
                                'latitude': float(event['latitude']),
                                'longitude': float(event['longitude']),
                                'event_id': idx,
                                'window_in_event': event_window_count
                            }
                            labels.append(label_dict)

                            metadata_dict = {
                                'earthquake_sample': earthquake_sample,
                                'window_start': start_sample + i,
                                'window_end': start_sample + i + window_samples,
                                'relative_to_earthquake': i - before_samples
                            }
                            metadata.append(metadata_dict)

                            event_window_count += 1

                    print(f"    ✅ {event_window_count}개 윈도우 생성")
                else:
                    print(f"    ⚠️ 이벤트 윈도우 너무 짧음: {len(event_window)} samples")
            else:
                print(f"    ⚠️ 이벤트가 데이터 범위를 벗어남")

        except Exception as e:
            print(f"    ❌ 이벤트 {idx+1} 처리 중 오류: {str(e)}")
            continue

    return np.array(windows), labels, metadata

# 지진 이벤트 기반 윈도우 생성
print("지진 이벤트 기반 윈도우 생성 시작...")

# 데이터 존재 여부 안전하게 확인
if combined_data is not None:
    print(f"✅ combined_data 준비됨: {combined_data.shape}")
else:
    print("❌ combined_data 없음")

try:
    if len(catalog) > 0:
        print(f"✅ catalog 준비됨: {len(catalog)}개 이벤트")
        catalog_ready = True
    else:
        print("❌ catalog 비어있음")
        catalog_ready = False
except:
    print("❌ catalog 문제 있음")
    catalog_ready = False

# 실제 윈도우 생성
if combined_data is not None and catalog_ready:
    print("윈도우 생성 함수 호출...")
    event_windows, event_labels, event_metadata = create_earthquake_windows(
        combined_data, catalog, sampling_rate, window_samples
    )
else:
    print("⚠️ 윈도우 생성 조건 미충족")
    event_windows = np.array([])
    event_labels = []
    event_metadata = []

    print(f"\n✅ 이벤트 기반 윈도우 생성 완료:")
    print(f"  📊 총 윈도우 수: {len(event_windows)}")
    if len(event_windows) > 0:
        print(f"  📊 윈도우 shape: {event_windows[0].shape}")
        print(f"  📊 전체 shape: {event_windows.shape}")

# ============================================================================
# 4단계: 배경 노이즈 윈도우 생성 (지진이 없는 구간)
# ============================================================================
print(f"\n🌊 4단계: 배경 노이즈 윈도우 생성")

def create_background_windows(combined_data, window_samples, overlap_samples,
                             exclude_ranges=None):
    """배경 노이즈 윈도우 생성 (지진이 없는 구간)"""

    background_windows = []
    background_metadata = []

    # 전체 데이터에서 지진 구간을 제외한 부분에서 윈도우 생성
    total_samples = len(combined_data)

    # 지진 구간 제외 (단순화를 위해 중앙 1/3 구간을 지진 구간으로 가정)
    exclude_start = total_samples // 3
    exclude_end = total_samples * 2 // 3

    print(f"배경 노이즈 구간:")
    print(f"  🔸 구간 1: 0 ~ {exclude_start} samples")
    print(f"  🔸 구간 2: {exclude_end} ~ {total_samples} samples")

    # 첫 번째 구간에서 윈도우 생성
    for i in range(0, exclude_start - window_samples, overlap_samples):
        window = combined_data[i:i + window_samples]
        if len(window) == window_samples:
            background_windows.append(window)
            background_metadata.append({
                'window_start': i,
                'window_end': i + window_samples,
                'type': 'background_1'
            })

    # 두 번째 구간에서 윈도우 생성
    for i in range(exclude_end, total_samples - window_samples, overlap_samples):
        window = combined_data[i:i + window_samples]
        if len(window) == window_samples:
            background_windows.append(window)
            background_metadata.append({
                'window_start': i,
                'window_end': i + window_samples,
                'type': 'background_2'
            })

    return np.array(background_windows), background_metadata

# 배경 노이즈 윈도우 생성
if combined_data is not None:
    background_windows, background_metadata = create_background_windows(
        combined_data, window_samples, overlap_samples
    )

    print(f"✅ 배경 노이즈 윈도우 생성 완료:")
    print(f"  📊 배경 윈도우 수: {len(background_windows)}")
    if len(background_windows) > 0:
        print(f"  📊 윈도우 shape: {background_windows[0].shape}")

# ============================================================================
# 5단계: 노이즈 제거용 데이터 쌍 생성
# ============================================================================
print(f"\n🎯 5단계: 노이즈 제거용 데이터 쌍 생성")

def add_realistic_noise(clean_windows, noise_level=0.1):
    """현실적인 노이즈를 clean 신호에 추가"""

    noisy_windows = []

    print(f"노이즈 추가 진행:")
    print(f"  🔊 노이즈 레벨: {noise_level}")

    for i, clean_window in enumerate(clean_windows):
        # 1. 가우시안 노이즈
        gaussian_noise = np.random.normal(0, noise_level * np.std(clean_window),
                                        clean_window.shape)

        # 2. 전력선 간섭 (60Hz)
        time_axis = np.arange(len(clean_window)) / sampling_rate
        power_line_noise = 0.05 * noise_level * np.sin(2 * np.pi * 60 * time_axis)

        # 3. 저주파 드리프트
        drift_noise = 0.02 * noise_level * np.sin(2 * np.pi * 0.1 * time_axis)

        # 각 채널에 다른 노이즈 특성 적용
        noisy_window = clean_window.copy()
        for ch in range(clean_window.shape[1]):
            channel_noise = (gaussian_noise[:, ch] +
                           power_line_noise * (0.5 + 0.5 * ch) +  # 채널별 다른 강도
                           drift_noise[:, ch] if len(drift_noise.shape) > 1
                           else np.broadcast_to(drift_noise, (len(drift_noise),)))
            noisy_window[:, ch] += channel_noise

        noisy_windows.append(noisy_window)

        if (i + 1) % 10 == 0 or i == len(clean_windows) - 1:
            print(f"    진행률: {i+1}/{len(clean_windows)} ({(i+1)/len(clean_windows)*100:.1f}%)")

    return np.array(noisy_windows)

# Clean 데이터로 이벤트 윈도우 사용
if 'event_windows' in locals() and len(event_windows) > 0:
    clean_data = event_windows
    noisy_data = add_realistic_noise(clean_data, noise_level=0.15)

    print(f"✅ 노이즈 제거용 데이터 쌍 생성 완료:")
    print(f"  📊 Clean 데이터: {clean_data.shape}")
    print(f"  📊 Noisy 데이터: {noisy_data.shape}")

    # 노이즈 추가 효과 확인
    print(f"  📈 노이즈 추가 효과:")
    print(f"    Clean 표준편차: {np.std(clean_data):.4f}")
    print(f"    Noisy 표준편차: {np.std(noisy_data):.4f}")
    print(f"    노이즈 비율: {(np.std(noisy_data) - np.std(clean_data))/np.std(clean_data)*100:.1f}%")

# ============================================================================
# 6단계: 최종 정규화 및 데이터셋 구성
# ============================================================================
print(f"\n📐 6단계: 최종 정규화 및 데이터셋 구성")

def normalize_data(data, method='z_score'):
    """데이터 정규화"""
    if method == 'z_score':
        mean = np.mean(data)
        std = np.std(data)
        normalized = (data - mean) / (std + 1e-8)  # 0으로 나누기 방지
    elif method == 'min_max':
        min_val = np.min(data)
        max_val = np.max(data)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)

    return normalized, {'mean': mean if method == 'z_score' else min_val,
                       'scale': std if method == 'z_score' else (max_val - min_val)}

# 데이터 정규화
if 'noisy_data' in locals() and 'clean_data' in locals():
    print("데이터 정규화 진행:")

    # 전체 데이터셋에 대해 통계 계산
    all_data = np.concatenate([noisy_data.flatten(), clean_data.flatten()])

    # Z-score 정규화
    normalized_noisy, norm_stats = normalize_data(noisy_data, method='z_score')
    normalized_clean, _ = normalize_data(clean_data, method='z_score')

    print(f"  ✅ Z-score 정규화 완료")
    print(f"    평균: {norm_stats['mean']:.6f}")
    print(f"    표준편차: {norm_stats['scale']:.6f}")
    print(f"    정규화 후 범위: {normalized_noisy.min():.3f} ~ {normalized_noisy.max():.3f}")

# ============================================================================
# 7단계: 학습/검증 데이터 분할
# ============================================================================
print(f"\n📚 7단계: 학습/검증 데이터 분할")

def split_dataset(X, y, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """데이터셋을 train/validation/test로 분할"""

    total_samples = len(X)

    # 인덱스 섞기
    indices = np.random.permutation(total_samples)

    # 분할 지점 계산
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))

    # 분할
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return {
        'train': {'X': X[train_idx], 'y': y[train_idx], 'indices': train_idx},
        'val': {'X': X[val_idx], 'y': y[val_idx], 'indices': val_idx},
        'test': {'X': X[test_idx], 'y': y[test_idx], 'indices': test_idx}
    }

# 데이터 분할
if 'normalized_noisy' in locals() and 'normalized_clean' in locals():
    # 시드 설정으로 재현 가능한 분할
    np.random.seed(42)

    dataset = split_dataset(normalized_noisy, normalized_clean,
                          train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)

    print(f"✅ 데이터 분할 완료:")
    for split_name, split_data in dataset.items():
        print(f"  📊 {split_name.upper()}: {len(split_data['X'])}개 샘플")
        print(f"    X shape: {split_data['X'].shape}")
        print(f"    y shape: {split_data['y'].shape}")

# ============================================================================
# 최종 결과 요약
# ============================================================================
print(f"\n🎉 딥러닝용 전처리 완료!")
print("="*60)

if 'dataset' in locals():
    print(f"📊 최종 데이터셋:")
    print(f"  🎯 작업: 지진파 노이즈 제거 (Denoising)")
    print(f"  📐 입력 차원: {dataset['train']['X'].shape[1:]} (시간 x 채널)")
    print(f"  📈 데이터 정규화: Z-score")
    print(f"  🔀 데이터 분할:")
    for split_name, split_data in dataset.items():
        ratio = len(split_data['X']) / (len(dataset['train']['X']) +
                                      len(dataset['val']['X']) +
                                      len(dataset['test']['X'])) * 100
        print(f"    - {split_name.upper()}: {len(split_data['X'])}개 ({ratio:.1f}%)")

    print(f"\n🚀 다음 단계: 딥러닝 모델 학습 준비 완료!")
    print(f"  💡 추천 모델: U-Net, Autoencoder, or Transformer-based denoiser")
    print(f"  📝 사용법:")
    print(f"    X_train = dataset['train']['X']")
    print(f"    y_train = dataset['train']['y']")
    print(f"    # 모델 학습 시작!")
else:
    print(f"❌ 일부 단계 실패 - 디버깅 필요")

print(f"\n✅ 딥러닝용 전처리 파이프라인 완료! 🎊")



print("🔄 새로운 정답데이터 강제 로딩 시작!")

# 1. 기존 catalog 변수 완전히 삭제
if 'catalog' in locals():
    del catalog
    print("✅ 기존 catalog 변수 삭제")

# 2. 새로운 Excel 파일 강제 로딩
import pandas as pd

try:
    print("📂 Excel 파일 읽기 중...")

    # Excel 파일 읽기 (정확한 구조로)
    raw_data = pd.read_excel("inCountryEarthquakeList_20060101_20250704.xlsx",
                            header=1,      # 2번째 행을 헤더로
                            skiprows=[2])  # 3번째 행 건너뛰기

    print(f"📊 원시 데이터 로딩: {len(raw_data)} 행")

    # 3. 유효한 데이터만 필터링
    print("🧹 데이터 정리 중...")

    # number와 origin_time이 있는 행만 유지
    valid_mask = raw_data['number'].notna() & raw_data['origin_time'].notna()
    catalog_clean = raw_data[valid_mask].copy()

    print(f"📊 유효 데이터: {len(catalog_clean)} 행")

    # 4. 데이터 타입 변환
    print("🔧 데이터 변환 중...")

    # 위도/경도 문자열 처리 ("36.85 N" → 36.85)
    catalog_clean['latitude'] = catalog_clean['latitude'].astype(str).str.replace(' N', '').str.replace(' S', '').astype(float)
    catalog_clean['longitude'] = catalog_clean['longitude'].astype(str).str.replace(' E', '').str.replace(' W', '').astype(float)

    # Excel 날짜 숫자를 datetime으로 변환
    catalog_clean['origin_time'] = pd.to_datetime(catalog_clean['origin_time'], origin='1899-12-30', unit='D')

    # 5. 최종 catalog 생성
    catalog = catalog_clean.reset_index(drop=True)

    print(f"🎉 새로운 정답데이터 로딩 완료!")
    print(f"  📊 총 이벤트: {len(catalog)}개")
    print(f"  📈 규모 범위: {catalog['magnitude'].min():.1f} - {catalog['magnitude'].max():.1f}")
    print(f"  📅 시간 범위: {catalog['origin_time'].min().date()} ~ {catalog['origin_time'].max().date()}")
    print(f"  🌍 위치 범위:")
    print(f"    위도: {catalog['latitude'].min():.1f}°N - {catalog['latitude'].max():.1f}°N")
    print(f"    경도: {catalog['longitude'].min():.1f}°E - {catalog['longitude'].max():.1f}°E")

    print(f"\n📋 첫 3개 이벤트:")
    display_cols = ['number', 'origin_time', 'magnitude', 'depth', 'latitude', 'longitude']
    print(catalog[display_cols].head(3))

    print(f"\n✅ 성공! 이제 {len(catalog)}개 이벤트로 전처리 가능!")

except FileNotFoundError:
    print("❌ 파일을 찾을 수 없습니다!")
    print("📂 현재 디렉토리 파일 확인:")
    import os
    for f in os.listdir('.'):
        if 'earthquake' in f.lower() or f.endswith('.xlsx'):
            print(f"  - {f}")

except Exception as e:
    print(f"❌ 오류 발생: {str(e)}")
    print("💡 다른 방법 시도:")
    print("  1. 파일명 확인")
    print("  2. Excel 라이브러리 설치: pip install openpyxl")
    print("  3. 커널 재시작 후 다시 시도")



print("🔍 CSV 파일 찾기")

import os

# 현재 디렉토리의 모든 CSV 파일 확인
print("📂 현재 디렉토리의 CSV 파일들:")
csv_files = []
for f in os.listdir('.'):
    if f.endswith('.csv') and 'earthquake' in f.lower():
        csv_files.append(f)
        print(f"  ✅ {f}")

if csv_files:
    # 가장 적합한 CSV 파일 선택
    target_csv = csv_files[0]  # 첫 번째 파일 사용
    print(f"\n📄 사용할 CSV 파일: {target_csv}")

    # CSV 파일 읽기
    import pandas as pd
    try:
        catalog = pd.read_csv(target_csv, skiprows=2)
        print(f"✅ CSV 읽기 성공: {len(catalog)} 행")

        # 첫 몇 행 확인
        print("\n📋 첫 5행:")
        print(catalog.head())

        # 유효한 데이터 개수 확인
        valid_count = len(catalog.dropna(subset=[catalog.columns[0]]))
        print(f"📊 유효한 데이터: {valid_count}개")

    except Exception as e:
        print(f"❌ CSV 읽기 실패: {e}")
else:
    print("❌ 지진 관련 CSV 파일을 찾을 수 없습니다")
    print("💡 파일명을 직접 확인해주세요")



    print("🔧 구글 코랩용 안전한 CSV 읽기")

def safe_read_earthquake_csv(filename):
    """구글 코랩에서 안전하게 CSV 읽기"""

    try:
        # 여러 방법으로 시도
        methods = [
            {"skiprows": 2, "encoding": "utf-8"},
            {"header": 1, "skiprows": [2], "encoding": "utf-8"},
            {"skiprows": 2, "encoding": "cp949"},
            {"skiprows": 2, "encoding": "euc-kr"},
            {"header": 0, "encoding": "utf-8"},
        ]

        for i, method in enumerate(methods, 1):
            try:
                print(f"  시도 {i}: {method}")
                df = pd.read_csv(filename, **method)

                # 유효한 데이터인지 확인
                if len(df) > 100:  # 1000개 이상이면 성공
                    print(f"    ✅ 성공: {len(df)} 행")

                    # 컬럼명 수정
                    if len(df.columns) >= 7:
                        df.columns = ['number', 'origin_time_str', 'magnitude', 'depth',
                                    'max_intensity', 'latitude_str', 'longitude_str'] + list(df.columns[7:])

                        # 데이터 변환
                        df['magnitude'] = pd.to_numeric(df['magnitude'], errors='coerce')
                        df['latitude'] = df['latitude_str'].astype(str).str.extract(r'([\d.]+)').astype(float)
                        df['longitude'] = df['longitude_str'].astype(str).str.extract(r'([\d.]+)').astype(float)

                        try:
                            df['origin_time'] = pd.to_datetime(df['origin_time_str'], errors='coerce')
                        except:
                            df['origin_time'] = pd.to_datetime('2022-01-01')

                        # 유효한 데이터만 필터링
                        df_clean = df.dropna(subset=['magnitude', 'latitude', 'longitude']).reset_index(drop=True)

                        print(f"    📊 정리 후: {len(df_clean)}개 유효한 이벤트")
                        print(f"    📈 규모 범위: {df_clean['magnitude'].min():.1f} - {df_clean['magnitude'].max():.1f}")

                        return df_clean

                else:
                    print(f"    ❌ 데이터 부족: {len(df)} 행")

            except Exception as e:
                print(f"    ❌ 실패: {str(e)[:50]}...")
                continue

        print("❌ 모든 방법 실패")
        return None

    except Exception as e:
        print(f"❌ 전체 실패: {e}")
        return None

# 실행
catalog = safe_read_earthquake_csv("inCountryEarthquakeList_2006-01-01_2025-07-04.csv")

if catalog is not None:
    print(f"\n🎉 성공! {len(catalog)}개 지진 이벤트 로딩 완료")
else:
    print(f"\n❌ CSV 읽기 실패 - 기본 데이터 사용")
    # 기본 데이터 생성
    catalog = pd.DataFrame({
        'number': [1, 2],
        'origin_time': pd.to_datetime(['2022-01-01T10:00:00', '2022-01-02T15:30:00']),
        'magnitude': [2.7, 3.5],
        'depth': [10.0, 8.0],
        'latitude': [36.123, 36.789],
        'longitude': [127.456, 128.123]
    })



import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("🚀 딥러닝용 지진파 전처리 시작!")
print("="*60)

# ============================================================================
# 1단계: 데이터 준비 및 검증
# ============================================================================
print("\n📋 1단계: 데이터 준비 및 검증")

# 전처리된 지진파 데이터 확인
print("전처리된 지진파 데이터:")
for channel_name, data_info in final_processed_data.items():
    print(f"  🔸 {channel_name}: {data_info['length']} samples @ {data_info['sampling_rate']}Hz")

# 카탈로그 데이터 확인
if 'catalog' in locals() and catalog is not None:
    print(f"\n카탈로그 데이터:")
    print(f"  📊 {len(catalog)}개 지진 이벤트")
    print(f"  📅 컬럼: {list(catalog.columns)}")
    if 'magnitude' in catalog.columns:
        print(f"  📈 규모 범위: {catalog['magnitude'].min()} - {catalog['magnitude'].max()}")
else:
    print("⚠️ 카탈로그 데이터 로딩 필요")

    # 실제 정답데이터 파일 읽기
    try:
        import pandas as pd

        # Excel 파일 읽기 (헤더는 2번째 행, 데이터는 4번째 행부터)
        raw_catalog = pd.read_excel("inCountryEarthquakeList_20060101_20250704.xlsx",
                                   header=1, skiprows=[2])  # 2번째 행을 헤더로, 3번째 행 건너뛰기

        # 데이터 정리
        # 유효한 데이터만 필터링 (number와 origin_time이 있는 행)
        valid_mask = raw_catalog['number'].notna() & raw_catalog['origin_time'].notna()
        catalog_clean = raw_catalog[valid_mask].copy()

        # 위도/경도 문자열 처리 ("36.85 N" -> 36.85)
        if 'latitude' in catalog_clean.columns:
            catalog_clean['latitude'] = catalog_clean['latitude'].astype(str).str.replace(' N', '').str.replace(' S', '').astype(float)
        if 'longitude' in catalog_clean.columns:
            catalog_clean['longitude'] = catalog_clean['longitude'].astype(str).str.replace(' E', '').str.replace(' W', '').astype(float)

        # origin_time을 datetime으로 변환 (Excel 숫자를 날짜로)
        # Excel의 숫자 날짜를 pandas datetime으로 변환
        catalog_clean['origin_time'] = pd.to_datetime(catalog_clean['origin_time'], origin='1899-12-30', unit='D')

        # 최종 카탈로그
        catalog = catalog_clean.reset_index(drop=True)

        print(f"✅ 정답데이터 로딩 완료!")
        print(f"  📊 총 {len(catalog)}개 지진 이벤트")
        print(f"  📅 컬럼: {list(catalog.columns)}")
        print(f"  📈 규모 범위: {catalog['magnitude'].min():.1f} - {catalog['magnitude'].max():.1f}")
        print(f"  📍 위치 범위:")
        print(f"    위도: {catalog['latitude'].min():.2f}°N - {catalog['latitude'].max():.2f}°N")
        print(f"    경도: {catalog['longitude'].min():.2f}°E - {catalog['longitude'].max():.2f}°E")
        print(f"  📅 시간 범위: {catalog['origin_time'].min()} ~ {catalog['origin_time'].max()}")

        print(f"\n첫 3개 이벤트:")
        print(catalog[['number', 'origin_time', 'magnitude', 'depth', 'latitude', 'longitude', 'location']].head(3))

    except Exception as e:
        print(f"❌ 정답데이터 로딩 실패: {str(e)}")
        # 기본 테스트 데이터 생성
        catalog = pd.DataFrame({
            'number': [1, 2],
            'origin_time': pd.to_datetime(['2022-01-01T10:00:00', '2022-01-02T15:30:00']),
            'magnitude': [2.7, 3.5],
            'depth': [10.0, 8.0],
            'latitude': [36.123, 36.789],
            'longitude': [127.456, 128.123],
            'location': ['테스트 지역 1', '테스트 지역 2']
        })
        print(f"  📊 테스트 데이터 생성: {len(catalog)}개 이벤트")

# ============================================================================
# 2단계: 3채널 데이터 결합
# ============================================================================
print(f"\n🔗 2단계: 3채널 데이터 결합")

# 컬럼 검증 및 수정 (1,664개 데이터인 경우)
if 'catalog' in locals() and len(catalog) > 100:
    print("🔍 대용량 카탈로그 데이터 컬럼 검증...")

    if 'magnitude' not in catalog.columns:
        print("⚠️ magnitude 컬럼 없음 - 컬럼 구조 문제 해결 중...")
        print(f"현재 컬럼: {list(catalog.columns)}")

        # 컬럼이 'Unnamed'로 되어있다면 수정
        if any('Unnamed' in str(col) for col in catalog.columns):
            if len(catalog.columns) >= 7:
                catalog.columns = ['number', 'origin_time_str', 'magnitude', 'depth', 'max_intensity', 'latitude_str', 'longitude_str'] + list(catalog.columns[7:])

                # 데이터 변환
                catalog['magnitude'] = pd.to_numeric(catalog['magnitude'], errors='coerce')
                catalog['depth'] = pd.to_numeric(catalog['depth'], errors='coerce')
                catalog['latitude'] = catalog['latitude_str'].astype(str).str.extract(r'([\d.]+)').astype(float)
                catalog['longitude'] = catalog['longitude_str'].astype(str).str.extract(r'([\d.]+)').astype(float)

                try:
                    catalog['origin_time'] = pd.to_datetime(catalog['origin_time_str'], errors='coerce')
                except:
                    catalog['origin_time'] = pd.to_datetime('2022-01-01')  # 기본값

                # 유효한 데이터만 필터링
                catalog = catalog.dropna(subset=['magnitude', 'latitude', 'longitude']).reset_index(drop=True)

                print(f"✅ 컬럼 구조 수정 완료!")
                print(f"  📊 처리된 이벤트: {len(catalog)}개")
                print(f"  📈 규모 범위: {catalog['magnitude'].min():.1f} - {catalog['magnitude'].max():.1f}")
    else:
        print("✅ 카탈로그 컬럼 구조 정상")

# 채널 순서 정의 (Z, 1, 2 순서로)
channel_order = ['vertical_BHZ', 'horizontal_1_BH1', 'horizontal_2_BH2']
combined_channels = []

print("채널 결합 진행:")
for channel_name in channel_order:
    if channel_name in final_processed_data:
        data = final_processed_data[channel_name]['data']
        combined_channels.append(data)
        print(f"  ✅ {channel_name}: {len(data)} samples 추가")
    else:
        print(f"  ❌ {channel_name}: 채널 없음")

if len(combined_channels) == 3:
    # (시간, 채널) 형태로 결합
    combined_data = np.column_stack(combined_channels)
    print(f"✅ 3채널 결합 완료: {combined_data.shape} (시간 x 채널)")

    # 기본 통계
    print(f"  📊 데이터 범위: {combined_data.min():.3f} ~ {combined_data.max():.3f}")
    print(f"  📊 평균: {np.mean(combined_data):.3f}")
    print(f"  📊 표준편차: {np.std(combined_data):.3f}")
else:
    print("❌ 3채널 결합 실패")
    combined_data = None

# ============================================================================
# 3단계: 시간 기준 윈도잉
# ============================================================================
print(f"\n⏰ 3단계: 시간 기준 윈도잉")

# 윈도우 파라미터 설정
sampling_rate = 40  # Hz
window_duration = 20  # 초
window_samples = int(window_duration * sampling_rate)  # 800 samples
overlap_ratio = 0.5  # 50% 겹침
overlap_samples = int(window_samples * overlap_ratio)

print(f"윈도우 설정:")
print(f"  🕐 윈도우 길이: {window_duration}초 ({window_samples} samples)")
print(f"  🔄 겹침: {overlap_ratio*100}% ({overlap_samples} samples)")

# 카탈로그 기반 윈도잉 함수
def create_earthquake_windows(combined_data, catalog, sampling_rate,
                             window_samples, before_seconds=10, after_seconds=10):
    """지진 이벤트 기준으로 윈도우 생성"""

    windows = []
    labels = []
    metadata = []

    print(f"\n지진 이벤트별 윈도우 생성:")

    # pandas DataFrame을 안전하게 순회
    for idx in range(len(catalog)):
        try:
            # iloc을 사용하여 안전하게 행 접근
            event = catalog.iloc[idx]
            magnitude = float(event['magnitude'])

            print(f"  📍 이벤트 {idx+1}: M{magnitude}")

            # 시간 정보 (실제로는 카탈로그의 시간과 지진파 데이터의 시간을 매칭해야 함)
            # 여기서는 예시로 데이터 중앙 부분을 지진 발생 시점으로 가정
            total_samples = len(combined_data)
            earthquake_sample = total_samples // 2  # 중앙점을 지진 발생으로 가정

            # 지진 전후 구간 계산
            before_samples = int(before_seconds * sampling_rate)
            after_samples = int(after_seconds * sampling_rate)

            start_sample = earthquake_sample - before_samples
            end_sample = earthquake_sample + after_samples

            # 유효 범위 확인
            if start_sample >= 0 and end_sample <= total_samples:
                event_window = combined_data[start_sample:end_sample]

                # 윈도우가 충분히 긴지 확인
                if len(event_window) >= window_samples:
                    # 이벤트 윈도우 내에서 슬라이딩 윈도우 생성
                    event_window_count = 0
                    overlap_samples = window_samples // 2  # 50% 겹침

                    for i in range(0, len(event_window) - window_samples + 1, overlap_samples):
                        window = event_window[i:i + window_samples]

                        if len(window) == window_samples:
                            windows.append(window)

                            # 안전하게 라벨 생성
                            label_dict = {
                                'magnitude': magnitude,
                                'depth': float(event['depth']),
                                'latitude': float(event['latitude']),
                                'longitude': float(event['longitude']),
                                'event_id': idx,
                                'window_in_event': event_window_count
                            }
                            labels.append(label_dict)

                            metadata_dict = {
                                'earthquake_sample': earthquake_sample,
                                'window_start': start_sample + i,
                                'window_end': start_sample + i + window_samples,
                                'relative_to_earthquake': i - before_samples
                            }
                            metadata.append(metadata_dict)

                            event_window_count += 1

                    print(f"    ✅ {event_window_count}개 윈도우 생성")
                else:
                    print(f"    ⚠️ 이벤트 윈도우 너무 짧음: {len(event_window)} samples")
            else:
                print(f"    ⚠️ 이벤트가 데이터 범위를 벗어남")

        except Exception as e:
            print(f"    ❌ 이벤트 {idx+1} 처리 중 오류: {str(e)}")
            continue

    return np.array(windows), labels, metadata

# 지진 이벤트 기반 윈도우 생성
print("지진 이벤트 기반 윈도우 생성 시작...")

# 데이터 존재 여부 안전하게 확인
if combined_data is not None:
    print(f"✅ combined_data 준비됨: {combined_data.shape}")
else:
    print("❌ combined_data 없음")

try:
    if len(catalog) > 0:
        print(f"✅ catalog 준비됨: {len(catalog)}개 이벤트")
        catalog_ready = True
    else:
        print("❌ catalog 비어있음")
        catalog_ready = False
except:
    print("❌ catalog 문제 있음")
    catalog_ready = False

# 실제 윈도우 생성
if combined_data is not None and catalog_ready:
    print("윈도우 생성 함수 호출...")
    event_windows, event_labels, event_metadata = create_earthquake_windows(
        combined_data, catalog, sampling_rate, window_samples
    )
else:
    print("⚠️ 윈도우 생성 조건 미충족")
    event_windows = np.array([])
    event_labels = []
    event_metadata = []

    print(f"\n✅ 이벤트 기반 윈도우 생성 완료:")
    print(f"  📊 총 윈도우 수: {len(event_windows)}")
    if len(event_windows) > 0:
        print(f"  📊 윈도우 shape: {event_windows[0].shape}")
        print(f"  📊 전체 shape: {event_windows.shape}")

# ============================================================================
# 4단계: 배경 노이즈 윈도우 생성 (지진이 없는 구간)
# ============================================================================
print(f"\n🌊 4단계: 배경 노이즈 윈도우 생성")

def create_background_windows(combined_data, window_samples, overlap_samples,
                             exclude_ranges=None):
    """배경 노이즈 윈도우 생성 (지진이 없는 구간)"""

    background_windows = []
    background_metadata = []

    # 전체 데이터에서 지진 구간을 제외한 부분에서 윈도우 생성
    total_samples = len(combined_data)

    # 지진 구간 제외 (단순화를 위해 중앙 1/3 구간을 지진 구간으로 가정)
    exclude_start = total_samples // 3
    exclude_end = total_samples * 2 // 3

    print(f"배경 노이즈 구간:")
    print(f"  🔸 구간 1: 0 ~ {exclude_start} samples")
    print(f"  🔸 구간 2: {exclude_end} ~ {total_samples} samples")

    # 첫 번째 구간에서 윈도우 생성
    for i in range(0, exclude_start - window_samples, overlap_samples):
        window = combined_data[i:i + window_samples]
        if len(window) == window_samples:
            background_windows.append(window)
            background_metadata.append({
                'window_start': i,
                'window_end': i + window_samples,
                'type': 'background_1'
            })

    # 두 번째 구간에서 윈도우 생성
    for i in range(exclude_end, total_samples - window_samples, overlap_samples):
        window = combined_data[i:i + window_samples]
        if len(window) == window_samples:
            background_windows.append(window)
            background_metadata.append({
                'window_start': i,
                'window_end': i + window_samples,
                'type': 'background_2'
            })

    return np.array(background_windows), background_metadata

# 배경 노이즈 윈도우 생성
if combined_data is not None:
    background_windows, background_metadata = create_background_windows(
        combined_data, window_samples, overlap_samples
    )

    print(f"✅ 배경 노이즈 윈도우 생성 완료:")
    print(f"  📊 배경 윈도우 수: {len(background_windows)}")
    if len(background_windows) > 0:
        print(f"  📊 윈도우 shape: {background_windows[0].shape}")

# ============================================================================
# 5단계: 노이즈 제거용 데이터 쌍 생성
# ============================================================================
print(f"\n🎯 5단계: 노이즈 제거용 데이터 쌍 생성")

def add_realistic_noise(clean_windows, noise_level=0.1):
    """현실적인 노이즈를 clean 신호에 추가"""

    noisy_windows = []

    print(f"노이즈 추가 진행:")
    print(f"  🔊 노이즈 레벨: {noise_level}")

    for i, clean_window in enumerate(clean_windows):
        # 1. 가우시안 노이즈
        gaussian_noise = np.random.normal(0, noise_level * np.std(clean_window),
                                        clean_window.shape)

        # 2. 전력선 간섭 (60Hz)
        time_axis = np.arange(len(clean_window)) / sampling_rate
        power_line_noise = 0.05 * noise_level * np.sin(2 * np.pi * 60 * time_axis)

        # 3. 저주파 드리프트
        drift_noise = 0.02 * noise_level * np.sin(2 * np.pi * 0.1 * time_axis)

        # 각 채널에 다른 노이즈 특성 적용
        noisy_window = clean_window.copy()
        for ch in range(clean_window.shape[1]):
            channel_noise = (gaussian_noise[:, ch] +
                           power_line_noise * (0.5 + 0.5 * ch) +  # 채널별 다른 강도
                           drift_noise[:, ch] if len(drift_noise.shape) > 1
                           else np.broadcast_to(drift_noise, (len(drift_noise),)))
            noisy_window[:, ch] += channel_noise

        noisy_windows.append(noisy_window)

        if (i + 1) % 10 == 0 or i == len(clean_windows) - 1:
            print(f"    진행률: {i+1}/{len(clean_windows)} ({(i+1)/len(clean_windows)*100:.1f}%)")

    return np.array(noisy_windows)

# Clean 데이터로 이벤트 윈도우 사용
if 'event_windows' in locals() and len(event_windows) > 0:
    clean_data = event_windows
    noisy_data = add_realistic_noise(clean_data, noise_level=0.15)

    print(f"✅ 노이즈 제거용 데이터 쌍 생성 완료:")
    print(f"  📊 Clean 데이터: {clean_data.shape}")
    print(f"  📊 Noisy 데이터: {noisy_data.shape}")

    # 노이즈 추가 효과 확인
    print(f"  📈 노이즈 추가 효과:")
    print(f"    Clean 표준편차: {np.std(clean_data):.4f}")
    print(f"    Noisy 표준편차: {np.std(noisy_data):.4f}")
    print(f"    노이즈 비율: {(np.std(noisy_data) - np.std(clean_data))/np.std(clean_data)*100:.1f}%")

# ============================================================================
# 6단계: 최종 정규화 및 데이터셋 구성
# ============================================================================
print(f"\n📐 6단계: 최종 정규화 및 데이터셋 구성")

def normalize_data(data, method='z_score'):
    """데이터 정규화"""
    if method == 'z_score':
        mean = np.mean(data)
        std = np.std(data)
        normalized = (data - mean) / (std + 1e-8)  # 0으로 나누기 방지
    elif method == 'min_max':
        min_val = np.min(data)
        max_val = np.max(data)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)

    return normalized, {'mean': mean if method == 'z_score' else min_val,
                       'scale': std if method == 'z_score' else (max_val - min_val)}

# 데이터 정규화
if 'noisy_data' in locals() and 'clean_data' in locals():
    print("데이터 정규화 진행:")

    # 전체 데이터셋에 대해 통계 계산
    all_data = np.concatenate([noisy_data.flatten(), clean_data.flatten()])

    # Z-score 정규화
    normalized_noisy, norm_stats = normalize_data(noisy_data, method='z_score')
    normalized_clean, _ = normalize_data(clean_data, method='z_score')

    print(f"  ✅ Z-score 정규화 완료")
    print(f"    평균: {norm_stats['mean']:.6f}")
    print(f"    표준편차: {norm_stats['scale']:.6f}")
    print(f"    정규화 후 범위: {normalized_noisy.min():.3f} ~ {normalized_noisy.max():.3f}")

# ============================================================================
# 7단계: 학습/검증 데이터 분할
# ============================================================================
print(f"\n📚 7단계: 학습/검증 데이터 분할")

def split_dataset(X, y, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """데이터셋을 train/validation/test로 분할"""

    total_samples = len(X)

    # 인덱스 섞기
    indices = np.random.permutation(total_samples)

    # 분할 지점 계산
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))

    # 분할
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return {
        'train': {'X': X[train_idx], 'y': y[train_idx], 'indices': train_idx},
        'val': {'X': X[val_idx], 'y': y[val_idx], 'indices': val_idx},
        'test': {'X': X[test_idx], 'y': y[test_idx], 'indices': test_idx}
    }

# 데이터 분할
if 'normalized_noisy' in locals() and 'normalized_clean' in locals():
    # 시드 설정으로 재현 가능한 분할
    np.random.seed(42)

    dataset = split_dataset(normalized_noisy, normalized_clean,
                          train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)

    print(f"✅ 데이터 분할 완료:")
    for split_name, split_data in dataset.items():
        print(f"  📊 {split_name.upper()}: {len(split_data['X'])}개 샘플")
        print(f"    X shape: {split_data['X'].shape}")
        print(f"    y shape: {split_data['y'].shape}")

# ============================================================================
# 최종 결과 요약
# ============================================================================
print(f"\n🎉 딥러닝용 전처리 완료!")
print("="*60)

if 'dataset' in locals():
    print(f"📊 최종 데이터셋:")
    print(f"  🎯 작업: 지진파 노이즈 제거 (Denoising)")
    print(f"  📐 입력 차원: {dataset['train']['X'].shape[1:]} (시간 x 채널)")
    print(f"  📈 데이터 정규화: Z-score")
    print(f"  🔀 데이터 분할:")
    for split_name, split_data in dataset.items():
        ratio = len(split_data['X']) / (len(dataset['train']['X']) +
                                      len(dataset['val']['X']) +
                                      len(dataset['test']['X'])) * 100
        print(f"    - {split_name.upper()}: {len(split_data['X'])}개 ({ratio:.1f}%)")

    print(f"\n🚀 다음 단계: 딥러닝 모델 학습 준비 완료!")
    print(f"  💡 추천 모델: U-Net, Autoencoder, or Transformer-based denoiser")
    print(f"  📝 사용법:")
    print(f"    X_train = dataset['train']['X']")
    print(f"    y_train = dataset['train']['y']")
    print(f"    # 모델 학습 시작!")
else:
    print(f"❌ 일부 단계 실패 - 디버깅 필요")

print(f"\n✅ 딥러닝용 전처리 파이프라인 완료! 🎊")



## 3. 결과데이터 파일저장

import pandas as pd
import numpy as np

print("📋 딥러닝 전처리 결과를 CSV로 저장")
print("="*50)

def save_earthquake_data_to_csv(dataset):
    """딥러닝 전처리된 지진파 데이터를 CSV로 저장"""

    def reshape_and_save(data_X, data_y, filename_prefix):
        """3D 지진파 데이터를 2D CSV로 변환하여 저장"""

        print(f"💾 {filename_prefix} 세트 저장 중...")

        n_samples, n_time, n_channels = data_X.shape
        print(f"  📊 형태: {n_samples}개 샘플 × {n_time}시점 × {n_channels}채널")

        # === X 데이터 (노이즈 있는 데이터) 저장 ===
        # (1664, 800, 3) → (1664, 2400) 형태로 변환
        X_reshaped = data_X.reshape(n_samples, -1)

        # 컬럼명 생성: t0_ch0, t0_ch1, t0_ch2, t1_ch0, t1_ch1, t1_ch2, ...
        X_columns = []
        for t in range(n_time):
            for ch in range(n_channels):
                channel_name = ['BHZ', 'BH1', 'BH2'][ch]  # 실제 채널명 사용
                X_columns.append(f't{t}_{channel_name}')

        # DataFrame 생성 및 저장
        X_df = pd.DataFrame(X_reshaped, columns=X_columns)
        X_df.to_csv(f'{filename_prefix}_X_noisy.csv', index=False)
        print(f"  ✅ {filename_prefix}_X_noisy.csv 저장완료 ({X_df.shape})")

        # === y 데이터 (깨끗한 데이터) 저장 ===
        y_reshaped = data_y.reshape(n_samples, -1)

        # 같은 컬럼명 사용
        y_columns = X_columns  # 동일한 구조

        y_df = pd.DataFrame(y_reshaped, columns=y_columns)
        y_df.to_csv(f'{filename_prefix}_y_clean.csv', index=False)
        print(f"  ✅ {filename_prefix}_y_clean.csv 저장완료 ({y_df.shape})")

        return X_df.shape, y_df.shape

    # 각 데이터셋 저장
    total_saved = 0

    for split_name in ['train', 'val', 'test']:
        if split_name in dataset and len(dataset[split_name]['X']) > 0:
            X_shape, y_shape = reshape_and_save(
                dataset[split_name]['X'],
                dataset[split_name]['y'],
                split_name
            )
            total_saved += X_shape[0]
        else:
            print(f"⚠️ {split_name} 데이터가 없습니다.")

    return total_saved

# 메타데이터 저장
def save_metadata_csv(dataset):
    """데이터셋 메타데이터를 CSV로 저장"""

    print(f"\n📊 메타데이터 저장 중...")

    # 기본 정보
    metadata = []

    for split_name in ['train', 'val', 'test']:
        if split_name in dataset:
            split_data = dataset[split_name]
            metadata.append({
                'split': split_name,
                'samples': len(split_data['X']),
                'time_steps': split_data['X'].shape[1] if len(split_data['X']) > 0 else 0,
                'channels': split_data['X'].shape[2] if len(split_data['X']) > 0 else 0,
                'total_features': split_data['X'].shape[1] * split_data['X'].shape[2] if len(split_data['X']) > 0 else 0
            })

    # 전체 정보 추가
    total_samples = sum(len(dataset[split]['X']) for split in ['train', 'val', 'test'] if split in dataset)

    metadata.append({
        'split': 'TOTAL',
        'samples': total_samples,
        'time_steps': 800,
        'channels': 3,
        'total_features': 2400
    })

    # 설정 정보 추가
    settings = pd.DataFrame([
        {'parameter': 'sampling_rate', 'value': '40 Hz'},
        {'parameter': 'window_duration', 'value': '20 seconds'},
        {'parameter': 'normalization', 'value': 'Z-score'},
        {'parameter': 'noise_level', 'value': '15%'},
        {'parameter': 'original_events', 'value': '1664'},
        {'parameter': 'channels', 'value': 'BHZ, BH1, BH2'},
        {'parameter': 'data_split', 'value': '70% train, 20% val, 10% test'}
    ])

    # 저장
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv('dataset_metadata.csv', index=False)
    settings.to_csv('dataset_settings.csv', index=False)

    print(f"  ✅ dataset_metadata.csv 저장완료")
    print(f"  ✅ dataset_settings.csv 저장완료")

    return metadata_df

# 실행
if 'dataset' in locals():
    print(f"🎯 현재 데이터셋 상태:")
    for split_name in ['train', 'val', 'test']:
        if split_name in dataset:
            print(f"  📚 {split_name}: {len(dataset[split_name]['X'])}개 샘플")

    # CSV 저장 실행
    total_saved = save_earthquake_data_to_csv(dataset)
    metadata_df = save_metadata_csv(dataset)

    print(f"\n🎉 CSV 저장 완료!")
    print(f"  📊 총 저장된 샘플: {total_saved}개")
    print(f"  📄 생성된 파일들:")
    print(f"    - train_X_noisy.csv (노이즈 있는 훈련 데이터)")
    print(f"    - train_y_clean.csv (깨끗한 훈련 데이터)")
    print(f"    - val_X_noisy.csv (노이즈 있는 검증 데이터)")
    print(f"    - val_y_clean.csv (깨끗한 검증 데이터)")
    print(f"    - test_X_noisy.csv (노이즈 있는 테스트 데이터)")
    print(f"    - test_y_clean.csv (깨끗한 테스트 데이터)")
    print(f"    - dataset_metadata.csv (데이터셋 정보)")
    print(f"    - dataset_settings.csv (설정 정보)")

    # 파일 크기 확인
    import os
    print(f"\n📏 파일 크기:")
    csv_files = [
        'train_X_noisy.csv', 'train_y_clean.csv',
        'val_X_noisy.csv', 'val_y_clean.csv',
        'test_X_noisy.csv', 'test_y_clean.csv'
    ]

    for file in csv_files:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / 1024 / 1024
            print(f"  📄 {file}: {size_mb:.1f} MB")

    print(f"\n💡 사용법:")
    print(f"  # 데이터 불러오기")
    print(f"  train_X = pd.read_csv('train_X_noisy.csv')")
    print(f"  train_y = pd.read_csv('train_y_clean.csv')")
    print(f"  # 딥러닝 학습에 사용!")

else:
    print("❌ 'dataset' 변수를 찾을 수 없습니다.")
    print("💡 먼저 딥러닝 전처리를 완료해주세요.")



## 딥러닝용으로 변환
import pandas as pd
import numpy as np

print("🚀 CSV → 딥러닝 데이터 변환 시작!")

def csv_to_deeplearning_ready(csv_prefix_list=['train', 'val', 'test']):
    """CSV에서 딥러닝 준비 완료 데이터로 한 번에 변환"""

    dataset = {}

    for prefix in csv_prefix_list:
        print(f"🔄 {prefix} 데이터 로딩 중...")

        try:
            # CSV 파일 읽기
            X_file = f'{prefix}_X_noisy.csv'
            y_file = f'{prefix}_y_clean.csv'

            X_df = pd.read_csv(X_file)
            y_df = pd.read_csv(y_file)

            # 3D 변환: (샘플, 2400) → (샘플, 800, 3)
            n_samples = len(X_df)
            X_3d = X_df.values.reshape(n_samples, 800, 3)
            y_3d = y_df.values.reshape(n_samples, 800, 3)

            # 딕셔너리에 저장
            dataset[prefix] = {
                'X': X_3d,  # 노이즈 있는 데이터
                'y': y_3d   # 깨끗한 데이터
            }

            print(f"  ✅ {prefix}: {X_3d.shape} → {y_3d.shape}")

        except FileNotFoundError:
            print(f"  ❌ {prefix} 파일들을 찾을 수 없습니다.")
        except Exception as e:
            print(f"  ❌ {prefix} 처리 중 오류: {e}")

    return dataset

# 실행!
dl_dataset = csv_to_deeplearning_ready()

# 결과 확인
if dl_dataset:
    print(f"\n🎉 변환 완료!")
    for split_name, data in dl_dataset.items():
        print(f"  📊 {split_name}: {data['X'].shape} (노이즈) → {data['y'].shape} (깨끗함)")

    print(f"\n💡 사용법:")
    print(f"  train_X = dl_dataset['train']['X']")
    print(f"  train_y = dl_dataset['train']['y']")
    print(f"  # 이제 딥러닝 모델에 바로 사용 가능! 🚀")
else:
    print("❌ 변환 실패")
