def next_epoch(self, all_values, experiment):
    def to_filename(epoch):
        return f"{self.experiment.path_models}/epoch{epoch}.pt"

    trial = all_values['epoch']['trial'][-1]
    print(trial.params)

    # iterations = all_values['epoch']['iteration']
    # epoch = iterations[-1]
    # torch.save(self.experiment.model.state_dict(), to_filename(epoch))
    # filename_column = list(map(lambda x: to_filename(x),iterations))
    # df = pd.DataFrame(all_values['epoch'])
    # df['model_path'] = filename_column
    # df.to_csv(f"{self.experiment.path}/epoch_data.csv")


def next_batch(self, all_values, experiment):
    batch_data = all_values['batch']
    # path = f"{self.experiment.path}/batch_data.csv"
    df = pd.DataFrame(batch_data)
    # df.to_csv(path)


def valid_loss_decreased(self, all_values, experiment):
    pass


def value_changed(self, all_values, experiment):
    pass


def next_trial(self, all_values, experiment):
    pass

