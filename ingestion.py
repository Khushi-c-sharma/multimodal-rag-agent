import logging
import os
import sys
from datetime import datetime
import json
import zipfile

try:
    from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
    from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
    from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
    from adobe.pdfservices.operation.io.stream_asset import StreamAsset
    from adobe.pdfservices.operation.pdf_services import PDFServices
    from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
    from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
    from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_renditions_element_type import ExtractRenditionsElementType
except ImportError as e:
    logging.error("Failed to import Adobe PDF Services SDK modules. Ensure the SDK is installed.", exc_info=True)
    sys.exit(1)


from google.colab import userdata
client_id = userdata.get('PDF_SERVICES_CLIENT_ID')
client_secret = userdata.get('PDF_SERVICES_CLIENT_SECRET')

INPUT_FILE_PATH = "/data/qatar_test_doc.pdf"

class ExtractTextInfoFromPDF:
    def __init__(self):
        try:
            with open(INPUT_FILE_PATH, 'rb') as f:
                input_stream = f.read()

            if not client_id or not client_secret:
                logging.error(
                    'PDF Services credentials not found. Set env vars `PDF_SERVICES_CLIENT_ID` and `PDF_SERVICES_CLIENT_SECRET` or provide `pdfservices-api-credentials.json`.'
                )
                sys.exit(1)

            try:
                credentials = ServicePrincipalCredentials(client_id=str(client_id), client_secret=str(client_secret))
            except Exception:
                logging.exception('Error during credentials setup')
                sys.exit(1)

            # Creates a PDF Services instance
            pdf_services = PDFServices(credentials=credentials)

            # Creates an asset(s) from source file(s) and upload
            input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.PDF)

            # Create parameters for the job
            extract_params = ExtractPDFParams(
            elements_to_extract=[
                ExtractElementType.TEXT,
                ExtractElementType.TABLES
            ],
            elements_to_extract_renditions=[
                ExtractRenditionsElementType.FIGURES  # Adds image extraction
            ]
)

            # Creates a new job instance
            extract_pdf_job = ExtractPDFJob(input_asset=input_asset, extract_pdf_params=extract_params)

            # Submit the job and gets the job result
            location = pdf_services.submit(extract_pdf_job)
            pdf_services_response = pdf_services.get_job_result(location, ExtractPDFResult)

            # Get content from the resulting asset(s)
            result_asset: CloudAsset = pdf_services_response.get_result().get_resource()
            stream_asset: StreamAsset = pdf_services.get_content(result_asset)

            # Creates an output stream and copy stream asset's content to it
            output_file_path = self.create_output_file_path()
            inp = stream_asset.get_input_stream()
            if hasattr(inp, 'read'):
                out_bytes = inp.read()
            else:
                out_bytes = inp

            with open(output_file_path, 'wb') as file:
                file.write(out_bytes)

            # Open the zip we just wrote and parse structuredData.json safely
            try:
                archive = zipfile.ZipFile(output_file_path, 'r')
                with archive.open('structuredData.json') as jsonentry:
                    data = json.load(jsonentry)

                for element in data.get('elements', []):
                    if element.get('Path', '').endswith('/H1'):
                        print(element.get('Text'))
            except FileNotFoundError:
                logging.error('Output zip not found: %s', output_file_path)
            except KeyError:
                logging.error('structuredData.json not found inside the zip')
            except json.JSONDecodeError:
                logging.error('Failed to parse JSON from structuredData.json')


        except (ServiceApiException, ServiceUsageException, SdkException) as e:
            logging.exception(f'Exception encountered while executing operation: {e}')
        except Exception:
            logging.exception('Unexpected error encountered')

    # Generates a string containing a directory structure and file name for the output file
    @staticmethod
    def create_output_file_path() -> str:
        now = datetime.now()
        time_stamp = now.strftime("%Y-%m-%dT%H-%M-%S")
        os.makedirs("output/ExtractTextInfoFromPDF", exist_ok=True)
        return f"output/ExtractTextInfoFromPDF/extract{time_stamp}.zip"

if __name__ == "__main__":
    ExtractTextInfoFromPDF()