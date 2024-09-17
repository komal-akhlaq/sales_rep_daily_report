import os

try:
    import psycopg2
except:
    os.system('pip install -qqq psycopg2')
    import psycopg2
import csv
try:
    import pandas as pd
except:
    os.system('pip install pandas')
    import pandas as pd
import json

import os

os.system('pip install -qqq reportlab')

from datetime import datetime, timedelta

# defining the time
# Calculate the time range for the past 24 hours
end_time = datetime.now()

# Set the start time to 7 PM yesterday
start_time = end_time - timedelta(days=1)
start_time = start_time.replace(hour=13, minute=0, second=0)

# end time is 12 hours after start time
end_time = start_time + timedelta(hours=12)

# Convert times to string format for SQL
start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')

print(f"Start time: {start_time_str}")
print(f"End time: {end_time_str}")

"""# Reading from database"""

# reading clients.csv

import psycopg2
import csv

# Database connection parameters
db_params = {
    'dbname': 'd5pt3225ki095v',
    'user': 'uchk5knobsqvs7',
    'password': 'pb82e547f1beee9040983d54a568e419b3d91a76ea16d6aaedd49b5fb41f1bcfe',
    'host': 'ec2-23-20-93-193.compute-1.amazonaws.com',
    'port': '5432'
}

# SQL query template to fetch distinct client IDs and names for the past month
fetch_client_ids_query = """
SELECT DISTINCT c.id AS client_id, c.fullname AS client_name
FROM
(
    SELECT
        t.client_id
    FROM
        textmessage t
    JOIN
        employee e ON t.created_by = e.id
    WHERE
        t.created >= NOW() - INTERVAL '1 month'
    AND e.fullname = ANY(%s)

    UNION

    SELECT
        ol.client_id
    FROM
        openphone_log ol
    JOIN
        employee e ON ol.from_ = e.phone
    WHERE
        ol.created_at_parsed >= NOW() - INTERVAL '1 month'
    AND ol.direction = 'outgoing'
    AND e.fullname = ANY(%s)
) as combined
JOIN
    public.client c ON combined.client_id = c.id
ORDER BY client_id;
"""

def fetch_client_ids_and_names():
    connection = None
    cursor = None
    try:
        # Connect to the database
        connection = psycopg2.connect(**db_params)
        cursor = connection.cursor()

        # Execute the query to fetch distinct client IDs and names for the past month
        cursor.execute(fetch_client_ids_query, (employee_names, employee_names))

        # Fetch all client IDs and names
        client_data = cursor.fetchall()

        # Write the client IDs and names to a CSV file
        with open('clients.csv', mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['client_id', 'client_name'])  # Write header

            for row in client_data:
                csv_writer.writerow(row)

        print("Client IDs and names have been saved to clients.csv")

    except Exception as error:
        print(f"Error fetching client data: {error}")
    finally:
        # Close the cursor and connection
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# List of employee names
employee_names = ['Mukund Chopra','John Green', 'Sara Edward','Ryan Rehman','Travis Keane', 'Maida Adams', 'Moohi Ahmed','Waseem Zubair','Alina Victor']

# Run the function to fetch client IDs and names and save to CSV
fetch_client_ids_and_names()

# Employee Records

import psycopg2
import csv
from datetime import datetime, timedelta

# List of employee names
employee_names = ['Mukund Chopra','John Green', 'Sara Edward','Ryan Rehman','Omar Blake','Simon Sinek','Daniel Robinson', 'Moohi Ahmed','Waseem Zubair','Alina Victor']

# Database connection parameters
db_params = {
    'dbname': 'd5pt3225ki095v',
    'user': 'uchk5knobsqvs7',
    'password': 'pb82e547f1beee9040983d54a568e419b3d91a76ea16d6aaedd49b5fb41f1bcfe',
    'host': 'ec2-23-20-93-193.compute-1.amazonaws.com',
    'port': '5432'
}

# SQL query template to fetch the required records for the specified time range
fetch_records_query_template = f"""
(
    SELECT
        to_char(t.created, 'YYYY-MM-DD HH24:MI:SS') AS timestamp,
        'text_created' AS type,
        t.message AS message,
        t.client_id,
        e.fullname AS employee_name
    FROM
        textmessage t
    JOIN
        employee e ON t.created_by = e.id
    WHERE
        e.fullname = %s
        AND t.created BETWEEN '{start_time_str}' AND '{end_time_str}'
)
UNION ALL
(
    SELECT
        to_char(ol.created_at_parsed, 'YYYY-MM-DD HH24:MI:SS') AS timestamp,
        'call_created' AS type,
        NULL AS message,
        ol.client_id,
        e.fullname AS employee_name
    FROM
        openphone_log ol
    JOIN
        employee e ON ol.from_ = e.phone
    WHERE
        e.fullname = %s
        AND ol.created_at_parsed BETWEEN '{start_time_str}' AND '{end_time_str}'
        AND ol.direction = 'outgoing'
)
UNION ALL
(
    SELECT
        to_char(ol.completed_at_parsed, 'YYYY-MM-DD HH24:MI:SS') AS timestamp,
        'call_completed' AS type,
        NULL AS message,
        ol.client_id,
        e.fullname AS employee_name
    FROM
        openphone_log ol
    JOIN
        employee e ON ol.from_ = e.phone
    WHERE
        e.fullname = %s
        AND ol.completed_at_parsed BETWEEN '{start_time_str}' AND '{end_time_str}'
        AND ol.direction = 'outgoing'
)
ORDER BY
    client_id, timestamp;
"""

def fetch_and_save_records_to_csv():
    connection = None
    cursor = None
    try:
        # Connect to the database
        connection = psycopg2.connect(**db_params)
        cursor = connection.cursor()

        # Open a CSV file to write the results
        with open('employee_records.csv', mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['timestamp', 'type', 'message', 'client id', 'employee_name'])

            # Loop over each employee name and fetch records
            for name in employee_names:
                cursor.execute(fetch_records_query_template, (name, name, name))

                # Fetch all records
                records = cursor.fetchall()

                # Write each record to the CSV file
                for record in records:
                    csv_writer.writerow(record)

                # Print the number of records fetched for the current employee
                # print(f"Number of records fetched for {name}: {len(records)}")
            print("Employee records fetched")

    except Exception as error:
        print(f"Error fetching records: {error}")
    finally:
        # Close the cursor and connection
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# Run the function to fetch records and save to CSV
fetch_and_save_records_to_csv()

"""# Employee Progress (New)"""

import psycopg2
import csv

# Database connection parameters
db_params = {
    'dbname': 'd5pt3225ki095v',
    'user': 'uchk5knobsqvs7',
    'password': 'pb82e547f1beee9040983d54a568e419b3d91a76ea16d6aaedd49b5fb41f1bcfe',
    'host': 'ec2-23-20-93-193.compute-1.amazonaws.com',
    'port': '5432'
}



def run_query_and_save_to_csv(sql_query, csv_file_name):
    connection = None
    cursor = None
    try:
        # Connect to the database
        connection = psycopg2.connect(**db_params)
        cursor = connection.cursor()

        # Execute the query
        cursor.execute(sql_query)

        # Fetch all records
        records = cursor.fetchall()

        # Get column names
        column_names = [desc[0] for desc in cursor.description]

        # Open a CSV file to write the results
        with open(csv_file_name, mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write the column headers
            csv_writer.writerow(column_names)

            # Write the records
            csv_writer.writerows(records)

        print(f"Records saved to {csv_file_name}")

    except Exception as error:
        print(f"Error running query: {error}")
    finally:
        # Close the cursor and connection
        if cursor:
            cursor.close()
        if connection:
            connection.close()

sql_query = f"""
SELECT
    csp.id,
    csp.client_id,
    c.fullname,
    CASE
        WHEN csp.current_stage = 1 THEN 'Stage 1: Not Interested'
        WHEN csp.current_stage = 2 THEN 'Stage 2: Initial Contact'
        WHEN csp.current_stage = 3 THEN 'Stage 3: Requirement Collection'
        WHEN csp.current_stage = 4 THEN 'Stage 4: Property Touring'
        WHEN csp.current_stage = 5 THEN 'Stage 5: Property Tour and Feedback'
        WHEN csp.current_stage = 6 THEN 'Stage 6: Application and Approval'
        WHEN csp.current_stage = 7 THEN 'Stage 7: Post-Approval and Follow-Up'
        WHEN csp.current_stage = 8 THEN 'Stage 8: Commission Collection'
        WHEN csp.current_stage = 9 THEN 'Stage 9: Dead Stage'
        ELSE 'Unknown Stage'
    END AS stage_name,
    csp.current_stage,
    csp.created_on,
    c.assigned_employee,
    c.assigned_employee_name
FROM
    client_stage_progression csp
JOIN
    client c
ON
    csp.client_id = c.id
WHERE
    csp.created_on BETWEEN '{start_time_str}' AND '{end_time_str}'
ORDER BY
    csp.created_on;

"""

csv_file_name = 'progress.csv'

# Run the function to execute the query and save to CSV
run_query_and_save_to_csv(sql_query, csv_file_name)

df5 = pd.read_csv('progress.csv')

# drop duplicates based on client_id, stage_name and assigned_employee
df5 = df5.drop_duplicates(subset=['client_id', 'current_stage'])

# drop all rows where current_stage is 9
df5 = df5[df5['current_stage'] != 9]

def employee_record(name, df):
    temp = df[df['assigned_employee_name'] == name]

    # group the dataframe by 'assigned_employee_name' and calculate the minimum and maximum 'current_stage' and 'stage_name'
    result_df = temp.groupby('fullname')[['current_stage', 'stage_name']].agg(['min', 'max']).reset_index()

    # rename the columns
    result_df.columns = ['Client', 'min_current_stage', 'max_current_stage', 'min_stage_name', 'max_stage_name']

    # create a new column change, which is the difference between the maximum and minimum current_stage
    result_df['change'] = result_df['max_current_stage'] - result_df['min_current_stage']

    # new colum Previous stage such that f'{min_current_stage}: {min_stage_name.split(':')[1]}' and check if : is present in min_stage_name to split
    result_df['Previous Stage'] = result_df['min_current_stage'].astype(str) + ': ' + result_df['min_stage_name'].apply(lambda x: x.split(':')[1] if ':' in x else x)

    # Current stage such that f'{max_current_stage}: {max_stage_name.split(':')[1]}' and check if : is present in max_stage_name to split
    result_df['Current Stage'] = result_df['max_current_stage'].astype(str) + ': ' + result_df['max_stage_name'].apply(lambda x: x.split(':')[1] if ':' in x else x)

    # check if - in the stage name, split the string and get the first element
    result_df['Previous Stage'] = result_df['Previous Stage'].apply(lambda x: x.split(' - ')[0])

    # check if - in the stage name, split the string and get the first element
    result_df['Current Stage'] = result_df['Current Stage'].apply(lambda x: x.split(' - ')[0])

    # drop the columns min_current_stage, max_current_stage, min_stage_name and max_stage_name, change
    result_df = result_df.drop(columns=['min_current_stage', 'max_current_stage', 'min_stage_name', 'max_stage_name','change'])
    return result_df

"""# Creating appropriate data"""

client_ids = {}
# reading client names
df = pd.read_csv(f'clients.csv')

# creating a dictionary with client ids and names
for index, row in df.iterrows():
    client_ids[row['client_id']] = row['client_name']

df = pd.read_csv(f'employee_records.csv')

# dropping the message column as it is of no use
df.drop('message', axis=1, inplace=True)

# creting a timestamp column
df['time_stamp'] = pd.to_datetime(df['timestamp'])

# Iterate through the rows to calculate the duration
for i in range(len(df) - 1):
    if df.loc[i, 'type'] == 'call_created' and df.loc[i+1, 'type'] == 'call_completed':
        # Convert timestamps to Unix time for calculation
        start_unix = df.loc[i, 'time_stamp'].timestamp()
        end_unix = df.loc[i+1, 'time_stamp'].timestamp()

        # Calculate the duration in seconds
        duration = end_unix - start_unix
        df.loc[i, 'call_duration'] = duration

# create a column for client name
df['client_name'] = df['client id'].map(client_ids)

# Drop the rows with type 'call_completed' as they are not needed
df = df[df['type'] != 'call_completed']

# change rows where type is call_created and call duration is null to 0
df.loc[(df['type'] == 'call_created') & (df['call_duration'].isnull()), 'call_duration'] = 0

# add average of call duration of that employee where call duration is 0
for index, row in df.iterrows():
    if row['call_duration'] == 0:
        df.loc[index, 'call_duration'] = df[df['employee_name'] == row['employee_name']]['call_duration'].mean()

# drop the time_stamp column
df.drop('time_stamp', axis=1, inplace=True)

"""## Report generation (New)"""

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Constants
ASSIGNED_MINUTES = 480  # 8 hours in minutes
SECONDS_PER_MESSAGE = 5  # 5 seconds per message

def add_employee_report(employee_name, df, df5, elements):
    styles = getSampleStyleSheet()

    # Add employee report title
    elements.append(Paragraph(f'Report for {employee_name}', styles['Heading2']))
    elements.append(Spacer(1, 12))

    # Collect one-line answers
    total_calls = df[df['employee_name'] == employee_name]['call_duration'].count()
    if total_calls > 0:
        elements.append(Paragraph(f'Total Calls: {total_calls}', styles['Normal']))

    total_duration_seconds = df[df['employee_name'] == employee_name]['call_duration'].sum()
    total_duration_minutes = total_duration_seconds // 60
    total_duration_remaining_seconds = total_duration_seconds % 60

    if total_duration_minutes > 0 or total_duration_remaining_seconds > 0:
        elements.append(Paragraph(f'Total Call Duration: {int(total_duration_minutes)} minutes {int(total_duration_remaining_seconds)} seconds', styles['Normal']))

    total_messages = df[(df['employee_name'] == employee_name) & (df['type'] == 'text_created')].shape[0]
    if total_messages > 0:
        elements.append(Paragraph(f'Total Messages: {total_messages}', styles['Normal']))

    # Calculate total message time in seconds
    total_message_time_seconds = total_messages * SECONDS_PER_MESSAGE

    # Calculate total work time in seconds (call duration + message time)
    total_work_time_seconds = total_duration_seconds + total_message_time_seconds

    # Convert total work time into minutes and seconds
    total_work_time_minutes = total_work_time_seconds // 60
    total_work_time_remaining_seconds = total_work_time_seconds % 60

    # Add the number of minutes assigned (8 hours = 480 minutes)
    elements.append(Paragraph(f'Assigned Time: {ASSIGNED_MINUTES} minutes', styles['Normal']))
    elements.append(Paragraph(f'Total Work Time: {int(total_work_time_minutes)} minutes {int(total_work_time_remaining_seconds)} seconds', styles['Normal']))

    # Determine number of clients handled
    employee_calls_clients = df[(df['employee_name'] == employee_name) & (df['type'] == 'call_created')]['client_name'].dropna().unique()
    employee_messages_clients = df[(df['employee_name'] == employee_name) & (df['type'] == 'text_created')]['client_name'].dropna().unique()

    # Merge both client lists and get unique clients
    unique_clients = pd.Series(list(set(employee_calls_clients) | set(employee_messages_clients))).dropna().unique()
    num_clients = len(unique_clients)

    if num_clients > 0:
        elements.append(Paragraph(f'Number of Clients Handled: {num_clients}', styles['Normal']))

    elements.append(Spacer(1, 12))

    # Add employee-specific record table
    employee_df = employee_record(employee_name, df5)
    if not employee_df.empty:
        elements.append(Paragraph("Employee Records:", styles['Heading2']))

        # Convert the employee records DataFrame to a list of lists for the Table
        record_data = [employee_df.columns.tolist()] + employee_df.values.tolist()

        # Create the table with styles
        record_table = Table(record_data)
        record_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Bold for header row
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),  # White text for header
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),  # Grey background for header
        ]))
        elements.append(record_table)
        elements.append(Spacer(1, 12))

    # Add separator line
    elements.append(Paragraph('---------------------------------------------------------------------------------------------------------------------------------------', styles['Normal']))
    elements.append(Spacer(1, 12))

def generate_combined_pdf_report(df, df5):
    pdf_filename = 'combined_employee_report.pdf'
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    elements = []

    employee_names = df['employee_name'].unique()
    for employee_name in employee_names:
        add_employee_report(employee_name, df, df5, elements)

    doc.build(elements)

# Assuming df, df5, and progress are defined
generate_combined_pdf_report(df, df5)


# Loading data
df = pd.read_csv(f'employee_records.csv')

# Dropping the message column as it is of no use
df.drop('message', axis=1, inplace=True)

# Creating a timestamp column
df['time_stamp'] = pd.to_datetime(df['timestamp'])

# Iterate through the rows to calculate the duration
for i in range(len(df) - 1):
    if df.loc[i, 'type'] == 'call_created' and df.loc[i+1, 'type'] == 'call_completed':
        # Convert timestamps to Unix time for calculation
        start_unix = df.loc[i, 'time_stamp'].timestamp()
        end_unix = df.loc[i+1, 'time_stamp'].timestamp()

        # Calculate the duration in seconds
        duration = end_unix - start_unix
        df.loc[i, 'call_duration'] = duration

# Creating a column for client name
df['client_name'] = df['client id'].map(client_ids)

# Drop the rows with type 'call_completed' as they are not needed
df = df[df['type'] != 'call_completed']

# Change rows where type is call_created and call duration is null to 0
df.loc[(df['type'] == 'call_created') & (df['call_duration'].isnull()), 'call_duration'] = 0

# Add average of call duration of that employee where call duration is 0
for index, row in df.iterrows():
    if row['call_duration'] == 0:
        df.loc[index, 'call_duration'] = df[df['employee_name'] == row['employee_name']]['call_duration'].mean()

# Drop the time_stamp column
df.drop('time_stamp', axis=1, inplace=True)

"""**CSV Generation**"""

import psycopg2
import csv
from datetime import datetime, timedelta
import pandas as pd

# Database connection parameters
db_params = {
    'dbname': 'd5pt3225ki095v',
    'user': 'uchk5knobsqvs7',
    'password': 'pb82e547f1beee9040983d54a568e419b3d91a76ea16d6aaedd49b5fb41f1bcfe',
    'host': 'ec2-23-20-93-193.compute-1.amazonaws.com',
    'port': '5432'
}

def run_query_and_save_to_csv(sql_query, csv_file_name):
    connection = None
    cursor = None
    try:
        # Connect to the database
        connection = psycopg2.connect(**db_params)
        cursor = connection.cursor()

        # Execute the query
        cursor.execute(sql_query)

        # Fetch all records
        records = cursor.fetchall()

        # Get column names
        column_names = [desc[0] for desc in cursor.description]

        # Open a CSV file to write the results
        with open(csv_file_name, mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write the column headers
            csv_writer.writerow(column_names)

            # Write the records
            csv_writer.writerows(records)

        print(f"Records saved to {csv_file_name}")

    except Exception as error:
        print(f"Error running query: {error}")
    finally:
        # Close the cursor and connection
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# SQL query to fetch data for the past week starting from 6pm
sql_query = """
SELECT
    csp.id,
    csp.client_id,
    c.fullname,
    csp.stage_name,
    csp.current_stage,
    csp.created_on,
    c.assigned_employee,
    c.assigned_employee_name
FROM
    client_stage_progression csp
JOIN
    client c
ON
    csp.client_id = c.id
WHERE
    csp.created_on >= (
        CURRENT_DATE - INTERVAL '1 day' + INTERVAL '13 hours'
    )
ORDER BY
    csp.created_on;
"""

csv_file_name = 'progress.csv'

# Run the function to execute the query and save to CSV
run_query_and_save_to_csv(sql_query, csv_file_name)

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_name)

# Drop duplicates based on client_id and current_stage
df = df.drop_duplicates(subset=['client_id', 'current_stage'])

# Drop all rows where current_stage is 9 (Dead Stage)
df = df[df['current_stage'] != 9]

# Create a pivot table to count the number of clients in each stage for each employee
df = df.pivot_table(index='assigned_employee_name', columns='current_stage', aggfunc='size', fill_value=0)

# Add missing stages (columns) if they don't exist
for i in range(2, 10):
    if i not in df.columns:
        df[i] = 0

# Reorder the columns
df = df[sorted(df.columns)]

# Drop unnecessary columns (those representing individual stages)
df1 = df.copy()

# Create new columns based on the calculations
df1['From tour to potential application'] = df1[5] + df1[6]
df1['From potential application to submitted application'] = df1[7] + df1[8] + df1[9]

# Calculate conversion rates with checks to avoid incorrect 100% values
df1['Lead-to-tour conversion rate %'] = df1.apply(lambda x: min((x[4] / x[2]) * 100, 100) if x[2] > 0 else 0, axis=1)
df1['Tour-to-close conversion rate %'] = df1.apply(lambda x: min((x[7] / x[4]) * 100, 100) if x[4] > 0 else 0, axis=1)

# Rename columns
df1.rename(columns={2: 'Total leads', 3: 'From inquiry to receiving option', 4: 'From receiving options to going on a tour'}, inplace=True)

# Drop individual stage columns
df1.drop(columns=[5, 6, 7, 8, 9], inplace=True)

# List of employee names to filter
employee_names = ['Mukund Chopra','John Green', 'Sara Edward','Ryan Rehman','Omar Blake','Simon Sinek','Daniel Robinson', 'Moohi Ahmed','Waseem Zubair','Alina Victor']

# Filter the DataFrame to include only these employees
df1 = df1[df1.index.isin(employee_names)]

# Replace NaN with 0
df1.fillna(0, inplace=True)

# Save the processed DataFrame to a CSV file
df1.to_csv('Agent_progress.csv')

# Output the processed DataFrame for review
print(df1)

"""**LLM Integration for message generation**"""

os.system('pip install groq')

import os
import logging
from groq import Groq
import pandas as pd
try:
    from PyPDF2 import PdfReader
except ImportError:
    os.system('pip install -qqq PyPDF2')
    from PyPDF2 import PdfReader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the Groq API key securely (consider using environment variables)
api_key = os.getenv("GROQ_API_KEY", "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")

# Initialize the Groq API client with the API key
client = Groq(api_key=api_key)

def load_weekly_report(csv_path):
    """Load the weekly report from a CSV file."""
    try:
        logging.info(f"Loading weekly report from {csv_path}")
        return pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Error loading weekly report: {e}")
        return pd.DataFrame()

def extract_pdf_data(pdf_path):
    """Extract daily report data from the PDF file."""
    pdf_data = {}
    try:
        reader = PdfReader(pdf_path)
        logging.info(f"Extracting data from PDF: {pdf_path}")

        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extract_text()

            if "Report for" in text:
                lines = text.split('\n')
                current_rep = None
                for line in lines:
                    if "Report for" in line:
                        rep_name = line.split("Report for")[1].strip()
                        pdf_data[rep_name] = {}
                        current_rep = rep_name
                    if current_rep:
                        if "Total Calls:" in line:
                            pdf_data[current_rep]['calls'] = int(line.split("Total Calls:")[1].strip())
                        elif "Total Call Duration:" in line:
                            pdf_data[current_rep]['call_duration'] = line.split("Total Call Duration:")[1].strip()
                        elif "Total Messages:" in line:
                            pdf_data[current_rep]['texts'] = int(line.split("Total Messages:")[1].strip())
                        elif "Number of Clients Handled:" in line:
                            pdf_data[current_rep]['clients_dealing_with'] = int(line.split("Number of Clients Handled:")[1].strip())

        logging.info("PDF data extraction completed.")
    except Exception as e:
        logging.error(f"Error extracting PDF data: {e}")

    return pdf_data

def evaluate_sales_rep_performance(rep_name, total_leads, tour_conversion_rate, daily_info):
    """Evaluate sales rep performance using LLaMA and generate feedback."""
    calls = daily_info.get('calls', 0)
    texts = daily_info.get('texts', 0)
    call_duration = daily_info.get('call_duration', 'N/A')
    clients_dealing_with = daily_info.get('clients_dealing_with', 0)

    # Construct the prompt for LLaMA evaluation
    prompt = (
        f"Briefly summarize the sales performance of {rep_name} and suggest next steps. "
        f"Performance data: {total_leads} leads, {tour_conversion_rate:.2f}% tour conversion rate, "
        f"{calls} calls, {texts} texts, {call_duration} total call duration, {clients_dealing_with} clients handled, "
    )

    # Call the LLaMA model using the Groq API
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
        )

        dynamic_remarks = chat_completion.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in LLaMA evaluation: {e}")
        dynamic_remarks = "Unable to generate feedback at this time."

    # Construct the final feedback message
    message = (
        f"Hi {rep_name},\n\n"
        f"This week, you handled {total_leads:.1f} leads with a tour conversion rate of {tour_conversion_rate:.2f}%. "
        f"Daily Report: You made {calls} calls and sent {texts} texts, with a total call duration of {call_duration}. "
        f"Currently, you are handling {clients_dealing_with} clients.\n\n"
        f"**Constructive Feedback:**\n**{dynamic_remarks}**\n\n"
        "### Final Notes\n"
        "- *Company policy:* We are committed to coaching skill gaps up to three times if the effort is evident. "
        "However, we will not coach effort gaps and will consider replacing the resource if the will is not there.\n\n"
    )

    return message

def generate_feedback(weekly_data, daily_data):
    """Generate brief and to-the-point feedback for all sales reps."""
    feedback_messages = []

    if weekly_data.empty:
        logging.error("No weekly data available.")
        return feedback_messages

    for _, row in weekly_data.iterrows():
        rep_name = row.get('assigned_employee_name')
        total_leads = row.get('Total leads', 0)
        tour_conversion_rate = row.get('Lead-to-tour conversion rate %', 0)

        if not rep_name:
            logging.warning("Missing sales rep name in weekly data.")
            continue

        # Extract daily report details
        daily_info = daily_data.get(rep_name, {})

        # Evaluate and generate feedback
        feedback = evaluate_sales_rep_performance(rep_name, total_leads, tour_conversion_rate, daily_info)
        feedback_messages.append(feedback)

    return feedback_messages

# Load weekly report and daily report data
csv_path = 'Agent_progress.csv'
pdf_path = 'combined_employee_report.pdf'

weekly_report_df = load_weekly_report(csv_path)
daily_report_data = extract_pdf_data(pdf_path)

# Generate and print the feedback messages
feedback_list = generate_feedback(weekly_report_df, daily_report_data)
for feedback in feedback_list:
    print(feedback)