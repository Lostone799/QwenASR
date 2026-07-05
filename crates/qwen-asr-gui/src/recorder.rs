//! 麦克风录音模块 — 使用 cpal 进行跨平台音频采集
//!
//! 采集默认输入设备的音频，下混为单声道 f32，
//! 停止时返回原始采样率的数据，由调用方重采样到 16kHz。

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;
use std::sync::{Arc, Mutex};
use crate::sync_ext::safe_lock;

/// 麦克风录音器
pub struct MicRecorder {
    /// 已采集的单声道 f32 样本
    samples: Arc<Mutex<Vec<f32>>>,
    /// cpal 音频流（drop 即停止录音）
    stream: Option<cpal::Stream>,
    /// 设备原始采样率
    sample_rate: u32,
    /// 录音开始时间
    start: std::time::Instant,
}

impl MicRecorder {
    /// 启动录音，使用默认输入设备
    pub fn start() -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or("找不到麦克风设备")?;

        let supported_config = device
            .default_input_config()
            .map_err(|e| format!("无法获取麦克风配置: {}", e))?;

        let sample_rate = supported_config.sample_rate().0;
        let channels = supported_config.channels();
        let sample_format = supported_config.sample_format();
        let num_channels = channels as usize;

        let samples = Arc::new(Mutex::new(Vec::<f32>::new()));

        let err_callback = |err: cpal::StreamError| {
            crate::logger::log_error(&format!("录音流错误: {}", err));
        };

        let stream = match sample_format {
            SampleFormat::F32 => {
                let buf = samples.clone();
                device
                    .build_input_stream(
                        &supported_config.into(),
                        move |data: &[f32], _: &_| {
                            if let Ok(mut buf) = buf.lock() {
                                for chunk in data.chunks(num_channels) {
                                    let avg: f32 =
                                        chunk.iter().sum::<f32>() / chunk.len() as f32;
                                    buf.push(avg);
                                }
                            }
                        },
                        err_callback,
                        None,
                    )
                    .map_err(|e| format!("无法创建录音流: {}", e))?
            }
            SampleFormat::I16 => {
                let buf = samples.clone();
                device
                    .build_input_stream(
                        &supported_config.into(),
                        move |data: &[i16], _: &_| {
                            if let Ok(mut buf) = buf.lock() {
                                for chunk in data.chunks(num_channels) {
                                    let avg: f32 = chunk
                                        .iter()
                                        .map(|&s| s as f32 / 32768.0)
                                        .sum::<f32>()
                                        / chunk.len() as f32;
                                    buf.push(avg);
                                }
                            }
                        },
                        err_callback,
                        None,
                    )
                    .map_err(|e| format!("无法创建录音流: {}", e))?
            }
            SampleFormat::U16 => {
                let buf = samples.clone();
                device
                    .build_input_stream(
                        &supported_config.into(),
                        move |data: &[u16], _: &_| {
                            if let Ok(mut buf) = buf.lock() {
                                for chunk in data.chunks(num_channels) {
                                    let avg: f32 = chunk
                                        .iter()
                                        .map(|&s| (s as f32 - 32768.0) / 32768.0)
                                        .sum::<f32>()
                                        / chunk.len() as f32;
                                    buf.push(avg);
                                }
                            }
                        },
                        err_callback,
                        None,
                    )
                    .map_err(|e| format!("无法创建录音流: {}", e))?
            }
            _ => {
                return Err(format!(
                    "不支持的音频格式: {:?}",
                    sample_format
                ))
            }
        };

        stream
            .play()
            .map_err(|e| format!("无法开始录音: {}", e))?;

        Ok(Self {
            samples,
            stream: Some(stream),
            sample_rate,
            start: std::time::Instant::now(),
        })
    }

    /// 停止录音，返回 (单声道 f32 样本, 原始采样率)
    pub fn stop(mut self) -> (Vec<f32>, u32) {
        self.stream.take(); // drop stream → 停止采集
        let samples = safe_lock(&self.samples).clone();
        (samples, self.sample_rate)
    }

    /// 获取已录制时长（秒）
    pub fn elapsed_sec(&self) -> f32 {
        self.start.elapsed().as_secs_f32() as f32
    }

    /// 获取已采集的样本数
    pub fn sample_count(&self) -> usize {
        safe_lock(&self.samples).len()
    }

    /// 原始采样率
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// 返回内部样本缓冲区的 Arc 句柄，供实时识别线程读取。
    /// 录音停止后此 Arc 仍然有效，线程可读取最终累积的样本。
    pub fn samples_arc(&self) -> Arc<Mutex<Vec<f32>>> {
        self.samples.clone()
    }
}
