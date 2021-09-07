# wazo-cdr-ml-tool
The CDR service allows you to keep track of the activity going on your stack. It's a data source that could be valuable in numerous ways. In this article, we will see how to make value out of your CDR using machine learning.

## The CDR dataset  
Like all machine learning tasks, we need a so-called *dataset* which is in our case a dump of the CDR into a CSV file obtained via Portal.  
The CDR is a *tabular dataset*, meaning that we have rows or *occurrences*, and columns or *variables*. In the case of the CDR, each row is a call, and each column is an attribute of the calls.  
The columns are the following:  
- id | *integer* | Unique identifier  
- tenant_uuid | *string* | Unique identifier for the location  
- answered | *boolean* |  
- start | *datetime* | Date and time at which the call landed on the system  
- answer | *datetime* | Date and time at which the call was answered (empty if called wasn't answered)  
- end | *datetime* | Date and time at which the call hung up  
- destination_extension | *string* | Extension (internal id or phone number) of the destination  
- destination_name | *string* | Name of the destination  
- destination_internal_extension | *integer* | Internal extension of the destination  
- destination_internal_context | *string* | Internal context used  
- destination_user_uuid | *string* | Unique identifier of the reached user  
- destination_line_id | *integer* | User line reached  
- duration | *integer* | Call duration  
- call_direction | *string* | Direction of the call (internal, inbound, outbound)  
- requested_name | *string* | Dialed name  
- requested_extension | *string* | Dialed extension (internal id or phone number) of the requested agent  
- requested_context | *string* | Context corresponding of the requested number  
- requested_internal_extension | *integer* | Internal extension corresponding to the request  
- requested_internal_context | *string* | Internal context corresponding to the request  
- source_extension | *string* | Caller extension (internal id or phone number) of the source agent  
- source_name | *string* | Caller name  
- source_internal_extension | *string* | Caller Internal extension  
- source_internal_context | *integer* | Caller internal context  
- source_user_uuid | *string* | Caller uuid  
- source_line_id | *integer* | Id of the caller line  

# Create your task
You can create a new task by providing input features and providing output target. See examples in the [task](./tasks) folder

Then you can run pre-made model on this task that have already been optimised to deal with CDR data with
