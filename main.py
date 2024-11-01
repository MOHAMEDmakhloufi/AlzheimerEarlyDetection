from DataPreprocessing import DataPreprocessor
from DataScraping import DataScraper
from FeatureEngineering import FeatureEngineering

if __name__ == '__main__':

    data_scraper = DataScraper("/home/mohamed/PycharmProjects/AlzheimerEarlyDetection/scraped_dataset/cd_enhance")
    data_scraper.pipeline()

    data_preprocessor = DataPreprocessor(print_cost=True)
    data_preprocessor.pipeline_noise_reduction('test_scraped_dataset',
                                               'test_noise_reduction_dataset')
    data_preprocessor.pipeline_investigator_and_Patient_Audio_Separation('test_scraped_dataset',
                                                                        'test_noise_reduction_dataset')

    feature_engineering = FeatureEngineering(print_cost=True)
    feature_engineering.pipeline_dividing_audio_into_chunks('test_processed_dataset',
                                                         'test_chunks_dataset',
                                                            10)
    feature_engineering.pipeline_normalise_chunk_amplitude('test_chunks_dataset',
                                                           'test_normalized_chunks_amplitude_dataset')
    feature_engineering.pipeline_normalise_chunk_length('test_normalized_chunks_amplitude_dataset',
                                                        'test_normalized_chunks_length_dataset',
                                                        3000)
    feature_engineering.pipeline_combine_mfcc_images('test_normalized_chunks_length_dataset')
