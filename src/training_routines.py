import torch
from ignite.engine.engine import Engine


def create_supervised_trainer(model, optimizer, loss_fn=None,
                              device=None, non_blocking=False,
                              output_transform=lambda x, y, y_pred, loss: loss.item()):
    """Adapted code from Ignite-Library in order to allow for handling of graphs."""

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
                
        if device is not None:
            batch = batch.to(device, non_blocking=non_blocking)
        
        res_dict = model(batch)
        
        batch.y = torch.abs(batch.y + torch.normal(0, 0.25, size=batch.y.shape).to(device).double()) # add some noise to target values
        
        loss = loss_fn(res_dict, batch)
        
        loss.backward()

        optimizer.step()
        
        
        return output_transform(batch.x, batch, res_dict, loss)

    return Engine(_update)


def create_supervised_evaluator(model, loss_fn=None, metrics=None,
                                device=None, non_blocking=False,
                                pred_collector_function=None,
                                output_transform=lambda x, y, y_pred: (y_pred, y,)):
    """Adapted code from Ignite-Library in order to allow for handling of graphs."""

    metrics = metrics or {}

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            
            if device is not None:
                batch = batch.to(device, non_blocking=non_blocking)
            
            res_dict = model(batch)
            
            return output_transform(batch.x, batch, res_dict)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine