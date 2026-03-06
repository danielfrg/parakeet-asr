import sys
from pathlib import Path
from typing import Optional, List

import click

from .segment import AlignedSegment, generate_srt_from_segments


@click.command(
    help="""
Transcribe long audio files robustly using NVIDIA Parakeet-TDT ASR model.
"""
)
@click.argument(
    "input_files",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
)
@click.option(
    "--model",
    default="nvidia/parakeet-tdt-0.6b-v2",
    show_default=True,
)
@click.option("--format", type=click.Choice(["srt", "csv"]), default="srt")
@click.option(
    "--device",
    default=None,
    show_default=False,
)
def main(
    input_files: tuple[Path, ...],
    output_dir: Optional[Path],
    model: str,
    device: Optional[str],
    format: str,
):
    # Load torch here so the CLI is faster to respond to non run commands
    import torch
    from .model import ParakeetASR, AlignedSegment

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    click.echo(f"Using device: {device}")

    if not input_files:
        click.echo("Error: No input files provided.", err=True)
        sys.exit(1)

    valid_input_files = [
        input_file for input_file in input_files if input_file.is_file()
    ]
    ignored_directories = [
        input_file for input_file in input_files if input_file.is_dir()
    ]

    for ignored_dir in ignored_directories:
        click.echo(f"Warning: Ignoring directory '{ignored_dir}'.", err=True)

    if not valid_input_files:
        click.echo("Error: No valid input files provided.", err=True)
        sys.exit(1)

    click.echo(f"Processing {len(valid_input_files)} file(s)...")

    try:
        for i, input_file in enumerate(valid_input_files, 1):
            # Set output directory to input file's directory if not specified
            current_output_dir = (
                output_dir if output_dir is not None else input_file.parent
            )

            # Initialize the transcriber for each file (required)
            transcriber = ParakeetASR(
                model_name=model,
                device=device,
                chunk_duration_sec=120,
                overlap_duration_sec=15,
                nemo_log_level=50,
            )

            click.echo(
                f"[{i}/{len(valid_input_files)}] Starting transcription of '{input_file.name}'..."
            )

            try:
                segments: List[AlignedSegment] = transcriber.transcribe(input_file)

                if not segments:
                    click.echo(
                        f"Warning: No transcription result was generated for '{input_file.name}'.",
                        err=True,
                    )
                    continue

                if format == "srt":
                    save_segments_to_srt(
                        segments, current_output_dir / f"{input_file.stem}.srt"
                    )
                elif format == "csv":
                    save_segments_to_csv(
                        segments, current_output_dir / f"{input_file.stem}.csv"
                    )
                else:
                    click.echo(f"Invalid format: {format}", err=True)
                    sys.exit(1)

                click.echo(f"✓ Completed transcription of '{input_file.name}'")

            except FileNotFoundError as e:
                click.echo(f"Error processing '{input_file.name}': {e}", err=True)
                continue
            except IOError as e:
                click.echo(
                    f"Error processing audio file '{input_file.name}': {e}", err=True
                )
                continue
            except Exception as e:
                click.echo(f"Error processing '{input_file.name}': {e}", err=True)
                continue

    except ValueError as e:
        click.echo(f"Configuration Error: {e}", err=True)
        sys.exit(1)
    except RuntimeError as e:
        click.echo(f"Model or runtime error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
        raise e
        sys.exit(1)
    finally:
        # Ensure cleanup is called explicitly if the transcriber object exists
        if "transcriber" in locals():
            transcriber.cleanup()


def save_segments_to_srt(segments: List[AlignedSegment], output_path: Path):
    """Saves a list of AlignedSegments to an SRT file."""
    srt_content = generate_srt_from_segments(segments)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)


def save_segments_to_csv(segments: List[AlignedSegment], output_path: Path):
    """Saves a list of AlignedSegments to a CSV file."""
    import csv

    csv_headers = ["Start (s)", "End (s)", "Segment"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        for seg in segments:
            writer.writerow([f"{seg.start:.3f}", f"{seg.end:.3f}", seg.text])


if __name__ == "__main__":
    main()
