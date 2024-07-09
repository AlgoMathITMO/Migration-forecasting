from pandas import json_normalize
import pandas as pd
import numpy as np
import copy

from sdmetrics.column_pairs import CorrelationSimilarity, ContingencySimilarity
from sdmetrics.single_table import NewRowSynthesis
from sdmetrics.single_column import MissingValueSimilarity, TVComplement
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CopulaGANSynthesizer, TVAESynthesizer, CTGANSynthesizer

EPOCHS = 5000


def generate_synth_C(data: pd.DataFrame, generator=TVAESynthesizer, n_samples: int = 3000) -> tuple[pd.DataFrame, dict]:
    # sample data
    data = data.sample(frac=1)

    # create metedata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    metadata.update_column(column_name='saldo', sdtype='categorical')
    
    # initialize generator
    synthesizer = generator(metadata, epochs=EPOCHS,
                                   enforce_min_max_values=True, enforce_rounding=False
                                #    numerical_distributions={
                                        # 'amenities_fee': 'beta',
                                        # 'checkin_date': 'uniform'}
                                    )
    
    # train generator
    synthesizer.fit(data)

    ##############################################################################
    # Quality assesment with statistical significance
    ##############################################################################
    
    synth_metrics = {'corr_sim': [], 'TV': [], 'new_row': []}
    
    for i in range(100):
        synth_data = synthesizer.sample(num_rows=len(data))
        
        synth_metrics['corr_sim'].append(CorrelationSimilarity.compute(
                                                real_data=data[data.columns[:].values],
                                                synthetic_data=synth_data[data.columns[:].values],
                                                coefficient='Pearson')
                                        )
        
        synth_metrics['TV'].append(TVComplement.compute(
                                            real_data=data['saldo'],
                                            synthetic_data=synth_data['saldo'])
                                        )
        
        synth_metrics['new_row'].append(NewRowSynthesis.compute(
                                        real_data=data,
                                        synthetic_data=synth_data,
                                        metadata=metadata,
                                        numerical_match_tolerance=0.1,
                                        synthetic_sample_size=20)
                                    )
        
    synth_metrics_mean =  {'corr_sim': np.mean(synth_metrics['corr_sim']),
                           'TV': np.mean(synth_metrics['TV']),
                           'new_row': np.mean(synth_metrics['new_row'])}
    
    ##############################################################################
    
    synth_data = synthesizer.sample(num_rows=n_samples)
    
    return synth_data, synth_metrics_mean


def generate_synth_R(data: pd.DataFrame, generator=TVAESynthesizer, n_samples: int = 3000) -> tuple[pd.DataFrame, dict]:
    # sample data
    data = data.sample(frac=1)

    # create metedata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    
    # initialize generator
    synthesizer = generator(metadata, epochs=EPOCHS,
                                   enforce_min_max_values=True, enforce_rounding=False
                                #    numerical_distributions={
                                        # 'amenities_fee': 'beta',
                                        # 'checkin_date': 'uniform'}
                                    )
    
    # train generator
    synthesizer.fit(data)

    ##############################################################################
    # Quality assesment with statistical significance
    ##############################################################################
    
    synth_metrics = {'corr_sim': [], 'new_row': []}
    
    for i in range(100):
        synth_data = synthesizer.sample(num_rows=len(data))
        
        synth_metrics['corr_sim'].append(CorrelationSimilarity.compute(
                                                real_data=data[data.columns[:].values],
                                                synthetic_data=synth_data[data.columns[:].values],
                                                coefficient='Pearson')
                                        )
        
        synth_metrics['new_row'].append(NewRowSynthesis.compute(
                                        real_data=data,
                                        synthetic_data=synth_data,
                                        metadata=metadata,
                                        numerical_match_tolerance=0.1,
                                        synthetic_sample_size=20)
                                    )
        
    synth_metrics_mean =  {'corr_sim': np.mean(synth_metrics['corr_sim']),
                           'new_row': np.mean(synth_metrics['new_row'])}
    
    ##############################################################################
    
    synth_data = synthesizer.sample(num_rows=n_samples)
    
    return synth_data, synth_metrics_mean
