import numpy as np  
import pandas as pd  
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
import plotly.express as px


class FeatureImportance:

    """
    
    Extract & Plot the Feature Names & Importance Values from a Scikit-Learn Pipeline.
    
    The input is a Pipeline that starts with a ColumnTransformer & ends with a regression or classification model. 
    As intermediate steps, the Pipeline can have any number or no instances from sklearn.feature_selection.
    Note: 
    If the ColumnTransformer contains Pipelines and if one of the transformers in the Pipeline is adding completely new columns, 
    it must come last in the pipeline. For example, OneHotEncoder, MissingIndicator & SimpleImputer(add_indicator=True) add columns 
    to the dataset that didn't exist before, so there should come last in the Pipeline.
    
    
    Parameters
    ----------
    pipeline : a Scikit-learn Pipeline class where the a ColumnTransformer is the first element and model estimator is the last element
    verbose : a boolean. Whether to print all of the diagnostics. Default is False.
    
    Attributes
    __________
    column_transformer_features :  A list of the feature names created by the ColumnTransformer prior to any selectors being applied
    transformer_list : A list of the transformer names that correspond with the `column_transformer_features` attribute
    discarded_features : A list of the features names that were not selected by a sklearn.feature_selection instance.
    discarding_selectors : A list of the selector names corresponding with the `discarded_features` attribute
    feature_importance :  A Pandas Series containing the feature importance values and feature names as the index.    
    plot_importances_df : A Pandas DataFrame containing the subset of features and values that are actually displaced in the plot. 
    feature_info_df : A Pandas DataFrame that aggregates the other attributes. The index is column_transformer_features. The transformer column contains the transformer_list.
        value contains the feature_importance values. discarding_selector contains discarding_selectors & is_retained is a Boolean indicating whether the feature was retained.
    
    
    
    """
    def __init__(self, pipeline, verbose=False):
        self.pipeline = pipeline
        self.verbose = verbose
        self.discarded_features = None
        self.discarding_selectors = None
        #self.transformer_list = None
        #self.column_transformer_features = None


    def flatten_pipeline(self, p):
        """
        Converts a tree-like structure of Pipeline into a flat list of transformers

        Parameters
        ----------
        p Pipeline :  fitted Pipeline object

        Returns
        -------
        list : list of transformers
        """

        leafs = []
        def _get_leaf_nodes(node):
            try:
                children = node[1].steps
            except AttributeError:
                children = []
            if node is not None:
                if len(children) == 0:
                    leafs.append(node)
                for n in children:
                    _get_leaf_nodes(n)
        _get_leaf_nodes(p)
        return leafs

    def _get_feature_names_from_single_transformer(self, transformer, transformer_name, orig_feature_names):

        if self.verbose:
            print('----> orig_feature_names: ', orig_feature_names)
            print('\n------> substep: ', transformer, '\n')
            print('\n------> substep_name: ', transformer_name, '\n')


        if hasattr(transformer, 'get_feature_names'):

            if self.verbose:
                print('Has get feature names')

            if 'input_features' in transformer.get_feature_names.__code__.co_varnames:

                names = list(transformer.get_feature_names(orig_feature_names))

            else:

                names = list(transformer.get_feature_names())

            if self.verbose:
                print('Names from get_feature_names: ', names)

        elif hasattr(transformer, 'indicator_') and transformer.add_indicator:
            if self.verbose:
                print('Has indicator_')

            # is this transformer one of the imputers & did it call the MissingIndicator?

            missing_indicator_indices = transformer.indicator_.features_
            missing_indicators = [orig_feature_names[idx] + '_missing_flag' \
                                  for idx in missing_indicator_indices]
            names = orig_feature_names + missing_indicators

            if self.verbose:
                print('Names with indicators added: ', names)

        elif hasattr(transformer, 'features_'):
            if self.verbose:
                print('Has features_')
            # is this a MissingIndicator class?
            missing_indicator_indices = transformer.features_
            missing_indicators = [orig_feature_names[idx] + '_missing_flag' \
                                  for idx in missing_indicator_indices]

            names = missing_indicators

            if self.verbose:
                print('Names from features_: ', names)

        else:
            if self.verbose:
                print('Class does not add new features, moving on using original_feature_names')
            names = orig_feature_names

        if self.verbose:
            print('Finished checking for added features. Resulting names: ', names)


        # Now we check if there was a selection of features
        if not hasattr(transformer, 'get_support'):
            if self.verbose:
                print('Does not have get_support. Returning unmasked feature names')
        else:
            if self.verbose:
                print('Has get_support. This step selects features. I am going to check the mask')

            # Make sure the transformer is fitted
            check_is_fitted(transformer)

            # Get the support mask for this step
            mask = transformer.get_support()
            if self.verbose: print(mask)
            if self.verbose: print(names)

            # Check that the length of the mask is the same as the current feature length,
            # if not, raise an Error
            if len(mask) != len(names):
                raise RuntimeError('Mismatch in length of features and support from transformer!')

            # Make a dictionary with features and mask items
            feature_mask_dict = dict(zip(names, mask))

            # Update names according to the mask
            kept_features = [feature for feature, is_retained in feature_mask_dict.items() \
                        if is_retained]

            discarded_features = [feature for feature, is_retained in feature_mask_dict.items() \
                                  if not is_retained]

            # Record discarded features in the object level counter
            self.discarded_features.extend(discarded_features)
            self.discarding_selectors.extend([transformer_name] * len(discarded_features))

            if self.verbose:
                print(f'\t{len(kept_features)} retained, {len(discarded_features)} discarded')
                if len(discarded_features) > 0:
                    print('\n\tdiscarded_features:\n\n', discarded_features)

            names = kept_features
            
        return names

    def _get_feature_names_from_pipeline(self, transformer, transformer_name, orig_feature_names):
        # If the transformer is a Pipeline, we need to flatten it first
        if not isinstance(transformer, Pipeline):
            raise ValueError
        else:
            # Use a recursive function to get all transformers from the pipeline,
            # in the order they are going to be executed:
            flat_transformers = self.flatten_pipeline((transformer_name, transformer))

            # We start setting the feature names as the original feature names
            names = orig_feature_names

            # Walk the flattened list, checking for feature names using the dedicated function
            # and updating the feature names as we go
            for n, t in flat_transformers:
                names = self._get_feature_names_from_single_transformer(t, n, names)

            return names


    def get_feature_names(self, verbose=None):  

        """
        Get the column names from the a ColumnTransformer containing transformers & pipelines
        Parameters
        ----------
        verbose : a boolean indicating whether to print summaries. 
            default = False
        Returns
        -------
        a list of the correct feature names
        Note: 
        If the ColumnTransformer contains Pipelines and if one of the transformers in the Pipeline is adding completely new columns, 
        it must come last in the pipeline. For example, OneHotEncoder, MissingIndicator & SimpleImputer(add_indicator=True) add columns 
        to the dataset that didn't exist before, so there should come last in the Pipeline.
        Inspiration: https://github.com/scikit-learn/scikit-learn/issues/12525 
        """

        if self.discarded_features is None:
            self.discarded_features = []

        if self.discarding_selectors is None:
            self.discarding_selectors = []
            
        if self.verbose: print('''\n\n---------\nRunning get_feature_names\n---------\n''')

        column_transformer = self.pipeline[0]
        assert isinstance(column_transformer, ColumnTransformer), "Input isn't a ColumnTransformer"
        check_is_fitted(column_transformer)

        feature_names = []

        for i, transformer_item in enumerate(column_transformer.transformers_):

            transformer_name, transformer, orig_feature_names = transformer_item
            orig_feature_names = list(orig_feature_names)
            
            if self.verbose:
                print('\n\n', i, '. Transformer/Pipeline: ', transformer_name, ',', 
                      transformer.__class__.__name__, '\n')
                print('\tn_orig_feature_names:', len(orig_feature_names))

            if transformer == 'drop':
                # Record discarded features in the object level counter
                self.discarded_features.extend(orig_feature_names)
                self.discarding_selectors.extend([transformer_name] * len(orig_feature_names))

                continue
                
            if isinstance(transformer, Pipeline):
                # if pipeline, use self._get_feature_names_from_pipeline() to get feature names
                feature_names_from_this_step = self._get_feature_names_from_pipeline(transformer, transformer_name, orig_feature_names)
            else:
                # if not pipeline, use the regular function
                feature_names_from_this_step = self._get_feature_names_from_single_transformer(transformer, transformer_name, orig_feature_names)


            if self.verbose:
                print('\tn_features after this step:', len(feature_names_from_this_step))
                print('\tfeature names after this step:\n', feature_names_from_this_step)

            feature_names.extend(feature_names_from_this_step)
            #transformer_list.extend([transformer_name] * len(names))

        if self.verbose:
            print('Done with all column transformations. Output feature names: ', feature_names)
            print('------' * 15)


        # After we are done processing the ColumnTransformer, which has to be the first step of the Pipeline,
        # we continue with the remaining steps of the pipeline


        remaining_pipeline = self.pipeline.steps[1:]

        if self.verbose:
            print('------' * 15)
            print('Processing rest of pipeline...: ', remaining_pipeline)


        for i, step_item in enumerate(remaining_pipeline):

            step_name, step = step_item

            if self.verbose:
                print('------' * 5)
                print('\nStep ', i, ": ", step_name, ',', step.__class__.__name__, '\n')
                print('------' * 5)

            # Each step in the pipeline is going to be a regular transformer
            feature_names = self._get_feature_names_from_single_transformer(step_name, step, feature_names)

        if self.verbose:
            print('Done processing whole pipeline. Output feature names: ', feature_names)
            print('------' * 15)

        return feature_names

    
    def get_selected_features(self, verbose=None):
        raise DeprecationWarning("Use self.get_feature_names() instead")
        return None
        
        

    def get_feature_importance(self):
        
        """
        Creates a Pandas Series where values are the feature importance values from the model and feature names are set as the index. 
        
        This Series is stored in the `feature_importance` attribute.
        Returns
        -------
        A pandas Series containing the feature importance values and feature names as the index.
        
        """
        
        assert isinstance(self.pipeline, Pipeline), "Input isn't a Pipeline"

        #features = self.get_selected_features()
        features = self.get_feature_names()
             
        assert hasattr(self.pipeline[-1], 'feature_importances_'),\
            "The last element in the pipeline isn't an estimator with a feature_importances_ attribute"
        
        importance_values = self.pipeline[-1].feature_importances_
        
        assert len(features) == len(importance_values),\
            "The number of feature names & importance values doesn't match"
        
        feature_importance = pd.Series(importance_values, index=features)
        self.feature_importance = feature_importance
        
        # create feature_info_df
        column_transformer_df =\
            pd.DataFrame(dict(transformer=self.transformer_list),
                         index=self.column_transformer_features)

        discarded_features_df =\
            pd.DataFrame(dict(discarding_selector=self.discarding_selectors),
                         index=self.discarded_features)

        importance_df = self.feature_importance.rename('value').to_frame()

        self.feature_info_df = \
            column_transformer_df\
            .join([importance_df, discarded_features_df])\
            .assign(is_retained = lambda df: ~df.value.isna())        


        return feature_importance
        
    
    def plot(self, top_n_features=100, rank_features=True, max_scale=True, 
             display_imp_values=True, display_imp_value_decimals=1,
             height_per_feature=25, orientation='h', width=750, height=None, 
             str_pad_width=15, yaxes_tickfont_family='Courier New', 
             yaxes_tickfont_size=15):
        """
        Plot the Feature Names & Importances 
        Parameters
        ----------
        top_n_features : the number of features to plot, default is 100
        rank_features : whether to rank the features with integers, default is True
        max_scale : Should the importance values be scaled by the maximum value & mulitplied by 100?  Default is True.
        display_imp_values : Should the importance values be displayed? Default is True.
        display_imp_value_decimals : If display_imp_values is True, how many decimal places should be displayed. Default is 1.
        height_per_feature : if height is None, the plot height is calculated by top_n_features * height_per_feature. 
        This allows all the features enough space to be displayed
        orientation : the plot orientation, 'h' (default) or 'v'
        width :  the width of the plot, default is 500
        height : the height of the plot, the default is top_n_features * height_per_feature
        str_pad_width : When rank_features=True, this number of spaces to add between the rank integer and feature name. 
            This will enable the rank integers to line up with each other for easier reading. 
            Default is 15. If you have long feature names, you can increase this number to make the integers line up more.
            It can also be set to 0.
        yaxes_tickfont_family : the font for the feature names. Default is Courier New.
        yaxes_tickfont_size : the font size for the feature names. Default is 15.
        Returns
        -------
        plot
        """
        if height is None:
            height = top_n_features * height_per_feature
            
        # prep the data
        
        all_importances = self.get_feature_importance()
        n_all_importances = len(all_importances)
        
        plot_importances_df =\
            all_importances\
            .nlargest(top_n_features)\
            .sort_values()\
            .to_frame('value')\
            .rename_axis('feature')\
            .reset_index()
                
        if max_scale:
            plot_importances_df['value'] = \
                                plot_importances_df.value.abs() /\
                                plot_importances_df.value.abs().max() * 100
            
        self.plot_importances_df = plot_importances_df.copy()
        
        if len(all_importances) < top_n_features:
            title_text = 'All Feature Importances'
        else:
            title_text = f'Top {top_n_features} (of {n_all_importances}) Feature Importances'       
        
        if rank_features:
            padded_features = \
                plot_importances_df.feature\
                .str.pad(width=str_pad_width)\
                .values
            
            ranked_features =\
                plot_importances_df.index\
                .to_series()\
                .sort_values(ascending=False)\
                .add(1)\
                .astype(str)\
                .str.cat(padded_features, sep='. ')\
                .values

            plot_importances_df['feature'] = ranked_features
        
        if display_imp_values:
            text = plot_importances_df.value.round(display_imp_value_decimals)
        else:
            text = None

        # create the plot 
        
        fig = px.bar(plot_importances_df, 
                     x='value', 
                     y='feature',
                     orientation=orientation, 
                     width=width, 
                     height=height,
                     text=text)
        fig.update_layout(title_text=title_text, title_x=0.5) 
        fig.update(layout_showlegend=False)
        fig.update_yaxes(tickfont=dict(family=yaxes_tickfont_family, 
                                       size=yaxes_tickfont_size),
                         title='')
        fig.show()
