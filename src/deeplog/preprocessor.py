import numpy  as np
import pandas as pd
import torch
from tqdm import tqdm

class Preprocessor(object):
    """Preprocessor for loading data from standard data formats."""

    def __init__(
        self,
        length=None,
        timeout=None,
        NO_EVENT=-1337,
        add_eos=False,
        eos_token='<EOS>',
        EOS_ID=None,
        context=None,
    ):
        """Preprocessor for loading data from standard data formats.

            Parameters
            ----------
            length : int
                Number of events in context.

            timeout : float
                Maximum time between context event and the actual event in
                seconds.

            NO_EVENT : int, default=-1337
                ID of NO_EVENT event, i.e., event returned for context when no
                event was present. This happens in case of timeout or if an
                event simply does not have enough preceding context events.

            add_eos : bool, default=False
                If True, appends an EOS token to the end of each machine/block sequence
                before context generation.

            eos_token : string or int, default='<EOS>'
                Identifier used for the end-of-sequence token.

            EOS_ID : int, optional, default=-1338
                Integer ID for the end-of-sequence token. If explicitly specified,
                it configures the EOS token as this integer ID.

            context : int, optional
                Alias for length.
            """
        # Set context length
        if length is None and context is not None:
            length = context
        elif length is None and context is None:
            raise TypeError("Preprocessor requires 'length' (or 'context') argument.")

        self.context_length = length
        self.timeout        = timeout

        # Set no-event event
        self.NO_EVENT = NO_EVENT

        # Set EOS token configuration
        self.add_eos = add_eos
        self.EOS_ID = EOS_ID if EOS_ID is not None else -1338
        if EOS_ID is not None and eos_token == '<EOS>':
            self.eos_token = EOS_ID
        else:
            self.eos_token = eos_token

        # Set required columns
        self.REQUIRED_COLUMNS = {'timestamp', 'event', 'machine'}


    ########################################################################
    #                      General data preprocessing                      #
    ########################################################################

    def sequence(
        self,
        data,
        labels=None,
        verbose=False,
        mapping=None,
        add_eos=None,
        eos_token=None,
        EOS_ID=None,
    ):
        """Transform pandas DataFrame into DeepCASE sequences.

            Parameters
            ----------
            data : pd.DataFrame
                Dataframe to preprocess.

            labels : int or array-like of shape=(n_samples,), optional
                If a int is given, label all sequences with given int. If an
                array-like is given, use the given labels for the data in file.
                Note: will overwrite any 'label' data in input file.

            verbose : boolean, default=False
                If True, prints progress in transforming input to sequences.

            mapping : dict(), default=None, optional
                If mapping is provided, use given mapping. Any additional unique values
                will be appended to the given mapping. NO_EVENT may be present in the 
                provided mapping, but it is not required.

            add_eos : bool, optional, default=None
                If True, append an EOS token at the end of each machine/block sequence.
                If None, defaults to self.add_eos.

            eos_token : string or int, optional, default=None
                EOS token identifier. If None, defaults to self.eos_token.

            EOS_ID : int, optional, default=None
                Integer ID for the EOS token. If provided, overrides token identifier.

            Returns
            -------
            context : torch.Tensor of shape=(n_samples, context_length)
                Context events for each event in events.

            events : torch.Tensor of shape=(n_samples,)
                Events in data.

            labels : torch.Tensor of shape=(n_samples,)
                Labels will be None if no labels parameter is given, and if data
                does not contain any 'labels' column.

            mapping : dict()
                Mapping from new event_id to original event_id.
                Sequencing will map all events to a range from 0 to n_events.
                This is because event IDs may have large values, which is
                difficult for a one-hot encoding to deal with. Therefore, we map
                all Event ID values to a new value in that range and provide
                this mapping to translate back.
            """
        ################################################################
        #                  Transformations and checks                  #
        ################################################################

        # Resolve EOS parameters
        if add_eos is None:
            if eos_token is not None or EOS_ID is not None:
                add_eos = True
            else:
                add_eos = self.add_eos

        if eos_token is not None:
            chosen_eos_token = eos_token
        elif EOS_ID is not None:
            chosen_eos_token = EOS_ID
        else:
            chosen_eos_token = self.eos_token

        # Work on a copy of data to avoid mutating caller DataFrame in-place
        data = data.copy()

        # Case where a single label is given
        if isinstance(labels, int):
            # Set given label to all labels
            labels = np.full(data.shape[0], labels, dtype=int)

        # Transform labels to numpy array
        labels = np.asarray(labels)

        # Check if data contains required columns
        if set(data.columns) & self.REQUIRED_COLUMNS != self.REQUIRED_COLUMNS:
            raise ValueError(
                ".csv file must contain columns: {}"
                .format(list(sorted(self.REQUIRED_COLUMNS)))
            )

        # Check if labels is same shape as data
        if labels.ndim and labels.shape[0] != data.shape[0]:
            raise ValueError(
                "Number of labels: '{}' does not correspond with number of "
                "samples: '{}'".format(labels.shape[0], data.shape[0])
            )

        # Append EOS tokens if requested
        if add_eos:
            if labels.ndim:
                data['_deeplog_label_temp'] = labels

            # Ensure data is sorted by timestamp within machines
            data = data.sort_values(by='timestamp')

            # Identify last row of each machine to determine boundary timestamps
            last_rows = data.groupby('machine', as_index=False).last()
            last_rows['event'] = chosen_eos_token

            # Advance timestamp by delta without exceeding timeout
            delta = 1
            if self.timeout is not None and not np.isinf(self.timeout):
                if delta > self.timeout:
                    delta = self.timeout / 2.0 if self.timeout > 0 else 0

            # Increment timestamp respecting dtype
            last_rows['timestamp'] = last_rows['timestamp'] + delta

            # Combine and sort strictly by machine and timestamp
            data = pd.concat([data, last_rows], ignore_index=True)
            data.sort_values(by=['machine', 'timestamp'], kind='stable', inplace=True, ignore_index=True)

            if '_deeplog_label_temp' in data.columns:
                labels = data['_deeplog_label_temp'].values
                del data['_deeplog_label_temp']

        ################################################################
        #                          Map events                          #
        ################################################################

        # Create mapping of events
        if mapping is None:
            if add_eos:
                all_unique = pd.Series(data['event'].unique())
                raw_events = [e for e in all_unique if e != chosen_eos_token and e != self.NO_EVENT]
            else:
                raw_events = [e for e in data['event'].unique() if e != self.NO_EVENT]

            try:
                sorted_events = np.unique(raw_events)
            except TypeError:
                sorted_events = sorted(
                    raw_events,
                    key=lambda x: (0, x) if isinstance(x, (int, float, np.integer, np.floating)) else (1, str(x))
                )

            mapping = {
                i: event for i, event in enumerate(sorted_events)
            }
            
            # Check that NO_EVENT is not in raw events
            if self.NO_EVENT in raw_events:
                raise ValueError(
                    "NO_EVENT ('{}') is also a valid Event ID".format(self.NO_EVENT)
                )

            # Check that chosen_eos_token is not in raw events
            if add_eos and chosen_eos_token in raw_events:
                raise ValueError(
                    "EOS token ('{}') is also a valid Event ID".format(chosen_eos_token)
                )
            
            mapping[len(mapping)] = self.NO_EVENT
            if add_eos and chosen_eos_token not in mapping.values():
                mapping[len(mapping)] = chosen_eos_token
        
        # Use given mapping and add missing events
        else:
            mapping = mapping.copy()
            difference = [e for e in data['event'].unique() if e not in mapping.values()]
            max_key = len(mapping)
            for i, val in enumerate(difference):
                mapping[i+max_key] = val

            # Check that NO_EVENT is not in unseen events
            if self.NO_EVENT in difference:
                raise ValueError(
                    "NO_EVENT ('{}') is also a valid Event ID".format(self.NO_EVENT)
                )
            
            # Add NO_EVENT if it is not present in mapping
            if self.NO_EVENT not in mapping.values():
                mapping[len(mapping)] = self.NO_EVENT

            # Add chosen_eos_token if add_eos is True and not in mapping
            if add_eos and chosen_eos_token not in mapping.values():
                mapping[len(mapping)] = chosen_eos_token

        mapping_inverse = {v: k for k, v in mapping.items()}

        # Apply mapping
        data['event'] = data['event'].map(mapping_inverse)

        ################################################################
        #                      Initialise results                      #
        ################################################################

        # Set events as events
        events_arr = data['event'].values
        if not events_arr.flags.writeable:
            events_arr = events_arr.copy()
        events = torch.tensor(events_arr, dtype=torch.long)

        # Set context full of NO_EVENTs
        context = torch.full(
            size       = (data.shape[0], self.context_length),
            fill_value = mapping_inverse[self.NO_EVENT],
            dtype      = torch.long,
        )

        # Set labels if given
        if labels.ndim:
            labels = torch.tensor(labels, dtype=torch.long)
        # Set labels if contained in data
        elif 'label' in data.columns:
            labels_arr = data['label'].values
            if not labels_arr.flags.writeable:
                labels_arr = labels_arr.copy()
            labels = torch.tensor(labels_arr, dtype=torch.long)
        # Otherwise set labels to None
        else:
            labels = None

        ################################################################
        #                        Create context                        #
        ################################################################

        if not add_eos:
            # Sort data by timestamp
            data = data.sort_values(by='timestamp')
            machine_grouped = data.groupby('machine')
        else:
            # When add_eos=True, data is already grouped by machine and sorted by timestamp
            machine_grouped = data.groupby('machine', sort=False)

        # Add verbosity
        if verbose: machine_grouped = tqdm(machine_grouped, desc='Loading')

        # Group by machine
        for machine, events_ in machine_grouped:
            # Get indices, timestamps and events
            indices    = events_.index.values
            timestamps = events_['timestamp'].values
            events_    = events_['event'].values

            # Initialise context for single machine
            machine_context = np.full(
                (events_.shape[0], self.context_length),
                mapping_inverse[self.NO_EVENT],
                dtype = int,
            )

            # Loop over all parts of the context
            for i in range(self.context_length):

                # Compute time difference between context and event
                time_diff = timestamps[i+1:] - timestamps[:-i-1]
                # Check if time difference is larger than threshold
                timeout_mask = time_diff > self.timeout

                # Set mask to NO_EVENT
                machine_context[i+1:, self.context_length-i-1] = np.where(
                    timeout_mask,
                    mapping_inverse[self.NO_EVENT],
                    events_[:-i-1],
                )

            # Ensure machine_context is writable before converting to torch Tensor
            if not machine_context.flags.writeable:
                machine_context = machine_context.copy()
            machine_context = torch.tensor(machine_context, dtype=torch.long)

            # Ensure indices NumPy array is writable before indexing with PyTorch
            if not indices.flags.writeable:
                indices = indices.copy()

            # Add machine_context to context
            context[indices] = machine_context

        ################################################################
        #                        Return results                        #
        ################################################################

        # Return result
        return context, events, labels, mapping


    ########################################################################
    #                     Preprocess different formats                     #
    ########################################################################

    def csv(
        self,
        path,
        nrows=None,
        labels=None,
        verbose=False,
        mapping=None,
        add_eos=None,
        eos_token=None,
        EOS_ID=None,
    ):
        """Preprocess data from csv file.

            Note
            ----
            **Format**: The assumed format of a .csv file is that the first line
            of the file contains the headers, which should include
            ``timestamp``, ``machine``, ``event`` (and *optionally* ``label``).
            The remaining lines of the .csv file will be interpreted as data.

            Parameters
            ----------
            path : string
                Path to input file from which to read data.

            nrows : int, default=None
                If given, limit the number of rows to read to nrows.

            labels : int or array-like of shape=(n_samples,), optional
                If a int is given, label all sequences with given int. If an
                array-like is given, use the given labels for the data in file.
                Note: will overwrite any 'label' data in input file.

            verbose : boolean, default=False
                If True, prints progress in transforming input to sequences.

            mapping : dict(), default=None, optional
                If mapping is provided, use given mapping. Any additional unique values
                will be appended to the given mapping. NO_EVENT may be present in the 
                provided mapping, but it is not required.

            add_eos : bool, optional, default=None
                If True, append an EOS token at the end of each machine/block sequence.
                If None, defaults to self.add_eos.

            eos_token : string or int, optional, default=None
                EOS token identifier. If None, defaults to self.eos_token.

            EOS_ID : int, optional, default=None
                Integer ID for the EOS token. If provided, overrides token identifier.

            Returns
            -------
            events : torch.Tensor of shape=(n_samples,)
                Events in data.

            context : torch.Tensor of shape=(n_samples, context_length)
                Context events for each event in events.

            labels : torch.Tensor of shape=(n_samples,)
                Labels will be None if no labels parameter is given, and if data
                does not contain any 'labels' column.
            """
        # Read data from csv file into pandas dataframe
        data = pd.read_csv(path, nrows=nrows)

        # Transform to sequences and return
        return self.sequence(
            data,
            labels=labels,
            verbose=verbose,
            mapping=mapping,
            add_eos=add_eos,
            eos_token=eos_token,
            EOS_ID=EOS_ID,
        )


    def json(
        self,
        path,
        labels=None,
        verbose=False,
        mapping=None,
        add_eos=None,
        eos_token=None,
        EOS_ID=None,
    ):
        """Preprocess data from json file.

            Note
            ----
            json preprocessing will become available in a future version.

            Parameters
            ----------
            path : string
                Path to input file from which to read data.

            labels : int or array-like of shape=(n_samples,), optional
                If a int is given, label all sequences with given int. If an
                array-like is given, use the given labels for the data in file.
                Note: will overwrite any 'label' data in input file.

            verbose : boolean, default=False
                If True, prints progress in transforming input to sequences.

            mapping : dict(), default=None, optional
                If mapping is provided, use given mapping. Any additional unique values
                will be appended to the given mapping. NO_EVENT may be present in the 
                provided mapping, but it is not required.

            add_eos : bool, optional, default=None
                If True, append an EOS token at the end of each machine/block sequence.

            eos_token : string or int, optional, default=None
                EOS token identifier.

            EOS_ID : int, optional, default=None
                Integer ID for the EOS token.

            Returns
            -------
            events : torch.Tensor of shape=(n_samples,)
                Events in data.

            context : torch.Tensor of shape=(n_samples, context_length)
                Context events for each event in events.

            labels : torch.Tensor of shape=(n_samples,)
                Labels will be None if no labels parameter is given, and if data
                does not contain any 'labels' column.
            """
        raise NotImplementedError("Parsing '.json' not yet implemented.")


    def ndjson(
        self,
        path,
        labels=None,
        verbose=False,
        mapping=None,
        add_eos=None,
        eos_token=None,
        EOS_ID=None,
    ):
        """Preprocess data from ndjson file.

            Note
            ----
            ndjson preprocessing will become available in a future version.

            Parameters
            ----------
            path : string
                Path to input file from which to read data.

            labels : int or array-like of shape=(n_samples,), optional
                If a int is given, label all sequences with given int. If an
                array-like is given, use the given labels for the data in file.
                Note: will overwrite any 'label' data in input file.

            verbose : boolean, default=False
                If True, prints progress in transforming input to sequences.

            mapping : dict(), default=None, optional
                If mapping is provided, use given mapping. Any additional unique values
                will be appended to the given mapping. NO_EVENT may be present in the 
                provided mapping, but it is not required.

            add_eos : bool, optional, default=None
                If True, append an EOS token at the end of each machine/block sequence.

            eos_token : string or int, optional, default=None
                EOS token identifier.

            EOS_ID : int, optional, default=None
                Integer ID for the EOS token.

            Returns
            -------
            events : torch.Tensor of shape=(n_samples,)
                Events in data.

            context : torch.Tensor of shape=(n_samples, context_length)
                Context events for each event in events.

            labels : torch.Tensor of shape=(n_samples,)
                Labels will be None if no labels parameter is given, and if data
                does not contain any 'labels' column.
            """
        raise NotImplementedError("Parsing '.ndjson' not yet implemented.")


    def text(
        self,
        path,
        nrows=None,
        labels=None,
        verbose=False,
        mapping=None,
        add_eos=None,
        eos_token=None,
        EOS_ID=None,
    ):
        """Preprocess data from text file.

            Note
            ----
            **Format**: The assumed format of a text file is that each line in
            the text file contains a space-separated sequence of event IDs for a
            machine. I.e. for *n* machines, there will be *n* lines in the file.

            Parameters
            ----------
            path : string
                Path to input file from which to read data.

            nrows : int, default=None
                If given, limit the number of rows to read to nrows.

            labels : int or array-like of shape=(n_samples,), optional
                If a int is given, label all sequences with given int. If an
                array-like is given, use the given labels for the data in file.
                Note: will overwrite any 'label' data in input file.

            verbose : boolean, default=False
                If True, prints progress in transforming input to sequences.

            mapping : dict(), default=None, optional
                If mapping is provided, use given mapping. Any additional unique values
                will be appended to the given mapping. NO_EVENT may be present in the 
                provided mapping, but it is not required.

            add_eos : bool, optional, default=None
                If True, append an EOS token at the end of each machine/block sequence.
                If None, defaults to self.add_eos.

            eos_token : string or int, optional, default=None
                EOS token identifier. If None, defaults to self.eos_token.

            EOS_ID : int, optional, default=None
                Integer ID for the EOS token. If provided, overrides token identifier.

            Returns
            -------
            events : torch.Tensor of shape=(n_samples,)
                Events in data.

            context : torch.Tensor of shape=(n_samples, context_length)
                Context events for each event in events.

            labels : torch.Tensor of shape=(n_samples,)
                Labels will be None if no labels parameter is given, and if data
                does not contain any 'labels' column.
            """
        # Initialise data
        events     = list()
        machines   = list()

        # Open text file
        with open(path) as infile:

            # Loop over each line, i.e. machine
            for machine, line in enumerate(infile):

                # Break if machine >= nrows
                if nrows is not None and machine >= nrows: break

                # Extract events for each machine
                for event in map(int, line.split()):

                    # Add data
                    events  .append(event)
                    machines.append(machine)

        # Transform to pandas DataFrame
        data = pd.DataFrame({
            'timestamp': np.arange(len(events)), # Increasing order
            'event'    : events,
            'machine'  : machines,
        })

        # Transform to sequences and return
        return self.sequence(
            data,
            labels=labels,
            verbose=verbose,
            mapping=mapping,
            add_eos=add_eos,
            eos_token=eos_token,
            EOS_ID=EOS_ID,
        )


if __name__ == "__main__":
    ########################################################################
    #                               Imports                                #
    ########################################################################

    import argformat
    import argparse
    import os

    ########################################################################
    #                           Parse arguments                            #
    ########################################################################

    # Create Argument parser
    parser = argparse.ArgumentParser(
        description     = "Preprocessor: processes data from standard formats into DeepCASE sequences.",
        formatter_class = argformat.StructuredFormatter
    )

    # Add arguments
    parser.add_argument('file',                                  help='file      to preprocess')
    parser.add_argument('--write',                               help='file      to write output')
    parser.add_argument('--type',              default='auto'  , help="file type to preprocess (auto|csv|json|ndjson|t(e)xt)")
    parser.add_argument('--context', type=int, default=10      , help="size of context")
    parser.add_argument('--timeout', type=int, default=60*60*24, help="maximum time between context and event")
    parser.add_argument('--add_eos', action='store_true'       , help="append EOS token to each sequence")
    parser.add_argument('--eos_token',         default='<EOS>' , help="EOS token identifier")

    # Parse arguments
    args = parser.parse_args()

    ########################################################################
    #                              Parse type                              #
    ########################################################################

    # Allowed extensions
    ALLOWED_EXTENSIONS = {'csv', 'json', 'ndjson', 'txt', 'text'}

    # Infer type
    if args.type == 'auto':
        # Get file by extension
        args.type = os.path.splitext(args.file)[1][1:]
        # Check if recovered extension is allowed
        if args.type not in ALLOWED_EXTENSIONS:
            raise ValueError(
                "Automatically parsed extension not supported: '.{}'. "
                "Please manually specify --type (csv|json|ndjson|t(e)xt)"
                .format(args.type)
            )

    ########################################################################
    #                              Preprocess                              #
    ########################################################################

    # Create preprocessor
    preprocessor = Preprocessor(
        context   = args.context,
        timeout   = args.timeout,
        add_eos   = args.add_eos,
        eos_token = args.eos_token,
    )

    # Preprocess file
    if args.type == 'csv':
        events, context, labels = preprocessor.csv(args.file)
    elif args.type == 'json':
        events, context, labels = preprocessor.json(args.file)
    elif args.type == 'ndjson':
        events, context, labels = preprocessor.ndjson(args.file)
    elif args.type == 'txt' or args.type == 'text':
        events, context, labels = preprocessor.text(args.file)
    else:
        raise ValueError("Unsupported file type: '{}'".format(args.type))

    ########################################################################
    #                             Write output                             #
    ########################################################################

    # Write output if necessary
    if args.write:

        # Open output file
        with open(args.write, 'wb') as outfile:
            # Write output
            torch.save({
                'events' : events,
                'context': context,
                'labels' : labels,
            }, outfile)

        ####################################################################
        #                           Load output                            #
        ####################################################################

        # Open output file
        with open(args.write, 'rb') as infile:
            # Load output
            data = torch.load(infile)
            # Load variables
            events  = data.get('events')
            context = data.get('context')
            labels  = data.get('labels')

    ########################################################################
    #                             Show output                              #
    ########################################################################

    print("Events : {}".format(events))
    print("Context: {}".format(context))
    print("Labels : {}".format(labels))

