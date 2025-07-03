import nemo.collections.asr.models as asrmodels
import torch
import gc
from pathlib import Path
from pydub import AudioSegment
import os
import tempfile
from typing import List, Optional, Union
from difflib import SequenceMatcher
import logging

from nemo.utils import nemo_logging

from .segment import AlignedSegment


class ParakeetASR:
    """
    Transcribe audio files using NVIDIA NeMo's Parakeet-TDT model.
    It employs a sliding-window approach with segment merging for long audios.
    """

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v2",
        device: Optional[str] = None,
        chunk_duration_sec: float = 120.0,
        overlap_duration_sec: float = 15.0,
        nemo_log_level: int = logging.ERROR,
    ):
        """
        Initializes the ParakeetASR transcriber.

        Args:
            model_name (str): The name of the NeMo ASR model to use.
                              Defaults to "nvidia/parakeet-tdt-0.6b-v2".
            device (Optional[str]): The device to load the model on (e.g., "cuda", "cpu").
                                    If None, it will auto-detect (prefer CUDA).
            chunk_duration_sec (float): The duration of audio chunks in seconds for processing.
            overlap_duration_sec (float): The overlap duration between chunks in seconds
                                          to ensure smooth transitions and robust merging.
            nemo_log_level (str): The logging level for NeMo's internal messages
                                  (e.g., "INFO", "WARNING", "ERROR", "CRITICAL").
                                  Defaults to "ERROR" to suppress most output.
        """
        self.model_name = model_name
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.chunk_duration_sec = chunk_duration_sec
        self.overlap_duration_sec = overlap_duration_sec
        self.model: Optional[asrmodels.ASRModel] = None

        self.audio_path: Optional[Union[str, Path]] = None
        self.audio: Optional[AudioSegment] = None

        nemo_logger_instance = nemo_logging.Logger()
        nemo_logger_instance.set_verbosity(nemo_log_level)

    def load_model(self):
        """Loads the ASR model if not already loaded and sets NeMo's logging level."""
        if self.model is None:
            print(
                f"Loading NeMo Parakeet-TDT model '{self.model_name}' on {self.device}..."
            )

            try:
                self.model = asrmodels.ASRModel.from_pretrained(
                    model_name=self.model_name, map_location=self.device
                )
                self.model.eval()

                self.model.change_attention_model("rel_pos_local_attn", [256, 256])
                self.model.change_subsampling_conv_chunking_factor(1)
            except Exception as e:
                print(f"Error loading model '{self.model_name}': {e}")
                self.model = None
                raise

    def load_audio(self, audio_path: Union[str, Path]) -> AudioSegment:
        """
        Loads an audio file using pydub.

        Args:
            audio_path (Union[str, Path]): Path to the audio file to load.

        Returns:
            pydub.AudioSegment: The loaded audio as an AudioSegment object.

        Raises:
            FileNotFoundError: If the audio file doesn't exist.
            IOError: If there's an error loading the audio file.
        """
        self.audio_path = Path(audio_path)
        if not self.audio_path.exists():
            raise FileNotFoundError(f"Audio file not found at {self.audio_path}")

        print(f"Loading audio file '{audio_path.name}'")
        try:
            self.audio = AudioSegment.from_file(audio_path)
            print(f"Audio duration: {self.audio.duration_seconds:.2f} seconds")
            return self.audio
        except Exception as e:
            raise IOError(f"Error loading audio file {audio_path}: {e}")

    def transcribe(
        self, audio_path: Optional[Union[str, Path]] = None
    ) -> List[AlignedSegment]:
        """
        Transcribes the audio file using a sliding-window approach with
        segment merging for long audios.

        Args:
            audio_path (optional): Path to the audio file to transcribe.
                If None, the audio file must have been loaded using `load_audio` first.

        Returns:
            List[AlignedSegment]: A list of AlignedSegment objects containing
                                  the transcribed text with start and end times.
        """
        if self.audio is None:
            self.load_audio(audio_path)

        if self.model is None:
            self.load_model()

        if self.model is None:
            raise RuntimeError("ASR model failed to load. Cannot transcribe.")

        step_duration_ms = (self.chunk_duration_sec - self.overlap_duration_sec) * 1000
        if step_duration_ms <= 0:
            raise ValueError(
                "Chunk duration must be greater than overlap duration to make progress."
            )

        all_segments: List[AlignedSegment] = []
        total_audio_length_ms = len(self.audio)

        # Estimate num_chunks: Add 1 to ensure last partial chunk is included.
        num_chunks = (total_audio_length_ms + step_duration_ms - 1) // step_duration_ms

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Iterate up to and slightly past the total audio length to ensure the last segment is processed.
                # The min(end_ms, total_audio_length_ms) inside the loop handles exact trimming.
                for i, start_ms in enumerate(
                    range(
                        0,
                        total_audio_length_ms + int(step_duration_ms),
                        int(step_duration_ms),
                    )
                ):
                    end_ms = start_ms + (self.chunk_duration_sec * 1000)
                    chunk_audio = self.audio[
                        start_ms : min(end_ms, total_audio_length_ms)
                    ]

                    if not chunk_audio.duration_seconds > 0.1:
                        # Skip very short (near-empty) chunks
                        continue

                    chunk_path = os.path.join(temp_dir, f"chunk_{i:04d}.wav")

                    # Important: Ensure 16kHz mono for NeMo model input
                    chunk_audio = chunk_audio.set_frame_rate(16000).set_channels(1)
                    chunk_audio.export(chunk_path, format="wav")

                    print(
                        f"Transcribing chunk {i + 1}/{num_chunks} "
                        f"(starts at {start_ms / 1000:.2f}s, duration {chunk_audio.duration_seconds:.2f}s) ---"
                    )

                    output = self.model.transcribe(
                        [chunk_path], timestamps=True, return_hypotheses=False
                    )

                    if (
                        not output
                        or not output[0].timestamp
                        or "segment" not in output[0].timestamp
                    ):
                        print("Warning: No segments transcribed for this chunk.")
                        continue

                    # Convert NeMo output to our AlignedSegment format and adjust timestamps
                    chunk_offset_sec = start_ms / 1000
                    current_chunk_segments = [
                        AlignedSegment(
                            text=seg["segment"],
                            start=seg["start"] + chunk_offset_sec,
                            end=seg["end"] + chunk_offset_sec,
                        )
                        for seg in output[0].timestamp["segment"]
                    ]

                    print(
                        f"Merging results (current segments in master: {len(all_segments)})..."
                    )
                    all_segments = self.merge_segment_lists(
                        all_segments, current_chunk_segments
                    )

            except Exception as e:
                print(f"An error occurred during transcription loop: {e}")
                raise

        return all_segments

    def merge_segment_lists(
        self, list_a: List[AlignedSegment], list_b: List[AlignedSegment]
    ) -> List[AlignedSegment]:
        """
        Merges two lists of AlignedSegments based on the longest common subsequence of words
        within the segments in their overlapping region.
        """
        if not list_a:
            return list_b
        if not list_b:
            return list_a

        # Find the approximate start of the overlap in list_a
        overlap_start_time_b = list_b[0].start

        a_start_for_overlap_search_idx = 0
        # Iterate from the end of list_a backwards, or from the beginning forwards
        # to find a good starting point for `a_words`
        # We need a window from `list_a` that definitely contains potential overlap
        # A simple heuristic: start looking in list_a for segments that end after
        # `list_b`'s start minus some buffer (e.g., 2x overlap duration)
        for i, seg_a in enumerate(list_a):
            if seg_a.end >= (overlap_start_time_b - (self.overlap_duration_sec * 2)):
                a_start_for_overlap_search_idx = i
                break

        # If list_a ends significantly before list_b starts, they might not truly overlap
        if list_a[-1].end < overlap_start_time_b - (self.overlap_duration_sec * 0.75):
            print("Warning: Lists appear non-overlapping, concatenating.")
            return list_a + list_b

        # Extract words from the relevant portions for matching
        # Use a reasonable sized window from list_a to avoid excessive memory for very long audios
        a_words_for_match = " ".join(
            [s.text for s in list_a[a_start_for_overlap_search_idx:]]
        ).split()
        b_words_for_match = " ".join([s.text for s in list_b]).split()

        if not a_words_for_match or not b_words_for_match:
            print(
                "Warning: Empty word lists for merge comparison, falling back to time-based cutoff."
            )
            # Fallback to simple time-based cutoff
            cutoff_time = list_a[-1].end - (self.overlap_duration_sec / 2)
            merged = [s for s in list_a if s.end < cutoff_time]
            merged.extend([s for s in list_b if s.start >= cutoff_time])
            return merged

        # Use difflib to find the longest matching block of words
        matcher = SequenceMatcher(
            None, a_words_for_match, b_words_for_match, autojunk=False
        )
        match = matcher.find_longest_match(
            0, len(a_words_for_match), 0, len(b_words_for_match)
        )

        # If a good match is found, stitch the lists together
        if match.size > 3:  # Require a multi-word match for confidence
            # Find the segment in `list_a` that corresponds to the start of the match
            a_cut_segment_idx = a_start_for_overlap_search_idx  # Default to no cut if match is not found in a_words_for_match
            current_word_count_a = 0
            for i, seg_a in enumerate(list_a[a_start_for_overlap_search_idx:]):
                if (
                    current_word_count_a >= match.a
                ):  # match.a is index in a_words_for_match
                    a_cut_segment_idx = a_start_for_overlap_search_idx + i
                    break
                current_word_count_a += len(seg_a.text.split())

            # Find the segment in `list_b` that corresponds to the start of the match
            b_start_segment_idx = 0
            current_word_count_b = 0
            for i, seg_b in enumerate(list_b):
                if (
                    current_word_count_b >= match.b
                ):  # match.b is index in b_words_for_match
                    b_start_segment_idx = i
                    break
                current_word_count_b += len(seg_b.text.split())

            # Construct the merged list:
            # Take all segments from list_a up to (but not including) the segment where the match starts in list_a.
            # Then append all segments from list_b starting from the segment where the match starts in list_b.
            merged_segments = list_a[:a_cut_segment_idx]
            merged_segments.extend(list_b[b_start_segment_idx:])
            return merged_segments
        else:
            # Fallback: A simple time-based cutoff if no good match is found.
            print(
                "Warning: Low-confidence merge. No strong word-level match found. Using simple time-based cutoff."
            )
            # Cut list 'a' at the point of overlap_duration_sec / 2 before its end
            cutoff_time = list_a[-1].end - (self.overlap_duration_sec * 0.5)
            # Keep segments from list_a that end before the cutoff
            merged = [s for s in list_a if s.end < cutoff_time]
            # Add segments from list_b that start after the cutoff
            merged.extend([s for s in list_b if s.start >= cutoff_time])
            return merged

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        """Explicitly release model resources and restore NeMo logging level."""
        if self.model is not None:
            print("Cleaning up model resources...")
            del self.model
            self.model = None
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            print("Model resources cleaned.")
