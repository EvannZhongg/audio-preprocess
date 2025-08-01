import os
import csv

from pipeline.global_var import PipelineParam

logger = PipelineParam.logger


def append_to_report(report_path, podcast_name, episode_name, file_path, initial_duration, final_duration):
    """
    Appends a new row to the processing report CSV file.
    Creates the file and writes the header if it doesn't exist.
    """
    file_exists = os.path.isfile(report_path)
    retention_rate = (final_duration / initial_duration) * 100 if initial_duration > 0 else 0
    retention_rate = min(100.0, retention_rate)

    with open(report_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "PodcastName", 
                "EpisodeName", 
                "FilePath",
                "InitialDuration(s)", 
                "FinalDuration(s)", 
                "RetentionRate(%)"
            ])
        
        writer.writerow([
            podcast_name, 
            episode_name, 
            file_path,
            f"{initial_duration:.2f}", 
            f"{final_duration:.2f}", 
            f"{retention_rate:.2f}"
        ])


def update_stats(stats, step_name, list_before, list_after):
    """A helper function to calculate and update processing statistics for a given step."""
    count_before = len(list_before)
    # The list can be empty
    duration_before = sum(s["end"] - s["start"] for s in list_before) if list_before else 0
    
    count_after = len(list_after)
    duration_after = sum(s["end"] - s["start"] for s in list_after) if list_after else 0
    
    stats['steps'][step_name]['discarded_count'] = count_before - count_after
    stats['steps'][step_name]['discarded_duration'] = duration_before - duration_after


def print_processing_summary(stats, audio_name):
    """Prints a formatted summary of the audio processing statistics."""
    logger.info(f"--- Processing Summary for: {audio_name} ---")

    initial_count = stats['initial']['count']
    initial_duration = stats['initial']['duration']

    if initial_count == 0:
        logger.info("No initial segments found. Final Retention: 0 segments, 0.00s (0.00%)")
        logger.info("-------------------------------------------------")
        return

    logger.info(f"Initial: {initial_count} segments, {initial_duration:.2f}s total duration.")
    
    for step_name, data in stats['steps'].items():
        discarded_count = data['discarded_count']
        if discarded_count > 0:
            percentage_dropped = (data['discarded_duration'] / initial_duration) * 100
            logger.info(
                f" > Dropped by {step_name}: {discarded_count} segments "
                f"({data['discarded_duration']:.2f}s) - {percentage_dropped:.2f}% of initial."
            )

    final_count = stats['final']['count']
    final_duration = stats['final']['duration']
    retention_rate_count = (final_count / initial_count) * 100
    retention_rate_duration = (final_duration / initial_duration) * 100 if initial_duration > 0 else 0
    retention_rate_duration = min(100.0, retention_rate_duration)


    logger.info("-" * 20)
    logger.info(
        f"Final Retention: {final_count} / {initial_count} segments ({retention_rate_count:.2f}%)"
    )
    logger.info(
        f"Final Duration: {final_duration:.2f}s / {initial_duration:.2f}s ({retention_rate_duration:.2f}%)"
    )
    logger.info("--- End of Summary ---")


