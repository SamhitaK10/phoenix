import * as fs from "fs";
import * as path from "path";
import {
  type componentsV1,
  HttpError,
  type PhoenixClient,
} from "@arizeai/phoenix-client";
import { Command } from "commander";

import { createPhoenixClient, resolveDatasetId } from "../client";
import {
  getConfigErrorMessage,
  resolveConfig,
  validateConfig,
} from "../config";
import { assertDeletesEnabled, confirmOrExit } from "../confirm";
import {
  AuthRequiredError,
  ExitCode,
  getExitCodeForError,
  InvalidArgumentError,
} from "../exitCodes";
import { writeError, writeOutput, writeProgress } from "../io";
import { collectString } from "../optionParsers";
import {
  type DatasetExamplesData,
  formatDatasetExamplesOutput,
  formatDatasetUploadOutput,
  type OutputFormat,
} from "./formatDataset";
import { formatDatasetsOutput } from "./formatDatasets";
import type { CommonOptions, DeleteOptions } from "./options";

type Dataset = componentsV1["schemas"]["Dataset"];

/**
 * Options for `px dataset get <dataset-identifier>`. Without `--version`,
 * fetches examples from the latest version of the dataset.
 */
interface DatasetGetOptions extends CommonOptions<OutputFormat> {
  /**
   * `--file <path>`: Save output to file instead of stdout. When set, output
   * is always written as JSON regardless of `--format`.
   *
   * @example "./examples.json"
   */
  file?: string;
  /**
   * `--split <name>`: Filter examples by split name. Repeatable to include
   * multiple splits. Defaults to no filter (all splits).
   *
   * @example ["train"]
   */
  split?: string[];
  /**
   * `--version <id>`: Fetch examples from a specific dataset version instead
   * of the latest.
   *
   * @example "RGF0YXNldFZlcnNpb246MQ=="
   */
  version?: string;
}

/**
 * Options for `px dataset list`.
 */
interface DatasetListOptions extends CommonOptions<OutputFormat> {
  /**
   * `--limit <number>`: Maximum number of datasets to fetch. Defaults to 100.
   *
   * @example 50
   */
  limit?: number;
}

const UPLOAD_FILE_FORMATS = ["csv", "jsonl", "json", "arrow"] as const;

type UploadFileFormat = (typeof UPLOAD_FILE_FORMATS)[number];

/**
 * Content type the upload endpoint expects for each file format. JSON files
 * are converted to JSONL before upload, so they share its content type.
 */
const UPLOAD_CONTENT_TYPES: Record<UploadFileFormat, string> = {
  csv: "text/csv",
  jsonl: "application/jsonl",
  json: "application/jsonl",
  arrow: "application/x-pandas-pyarrow",
};

/**
 * Arrow IPC files start with these bytes; IPC streams, the only Arrow
 * format the upload endpoint reads, do not.
 */
const ARROW_IPC_FILE_MAGIC = "ARROW1";

/**
 * Options for `px dataset upload <file>`.
 */
interface DatasetUploadOptions extends CommonOptions<OutputFormat> {
  /**
   * `--name <name>`: Name of the dataset to create, or to append to with
   * `--append`.
   *
   * @example "support-questions"
   */
  name: string;
  /**
   * `--input-keys <columns...>`: Columns that make up each example's input.
   *
   * @example ["question", "context"]
   */
  inputKeys: string[];
  /**
   * `--output-keys <columns...>`: Columns that make up each example's
   * expected output.
   *
   * @example ["answer"]
   */
  outputKeys?: string[];
  /**
   * `--metadata-keys <columns...>`: Columns stored as each example's
   * metadata.
   *
   * @example ["source", "difficulty"]
   */
  metadataKeys?: string[];
  /**
   * `--split-key <column>`: Column whose values are each example's split
   * name or a JSON list of split names.
   *
   * @example "split"
   */
  splitKey?: string;
  /**
   * `--file-format <format>`: Format of the file — `csv`, `jsonl`, `json`, or
   * `arrow` — overriding detection from its extension.
   *
   * @example "jsonl"
   */
  fileFormat?: string;
  /**
   * `--append`: Add the examples to the dataset named by `--name`, creating
   * it if needed. Without it, the upload fails if the name is taken.
   *
   * @example true
   */
  append?: boolean;
}

function isUploadFileFormat(value: string): value is UploadFileFormat {
  return (UPLOAD_FILE_FORMATS as readonly string[]).includes(value);
}

/**
 * Resolve the upload file format from `--file-format`, falling back to the
 * file extension.
 * @param params.filePath - path of the file being uploaded
 * @param params.formatOverride - value of `--file-format`, if passed
 */
function resolveUploadFileFormat({
  filePath,
  formatOverride,
}: {
  filePath: string;
  formatOverride?: string;
}): UploadFileFormat {
  const candidateFormat = (
    formatOverride ?? path.extname(filePath).slice(1)
  ).toLowerCase();
  if (isUploadFileFormat(candidateFormat)) {
    return candidateFormat;
  }
  const supportedFormats = UPLOAD_FILE_FORMATS.join(", ");
  throw new InvalidArgumentError(
    formatOverride === undefined
      ? `Cannot detect the format of ${filePath} from its extension. Use --file-format with one of: ${supportedFormats}`
      : `Unsupported --file-format "${formatOverride}". Use one of: ${supportedFormats}`
  );
}

function isRowObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Convert a JSON file holding an array of row objects to JSONL, the line
 * format the upload endpoint accepts.
 */
function convertJsonRowsToJsonl(fileContent: string): string {
  let parsedContent: unknown;
  try {
    // JSON.parse rejects a leading byte order mark.
    parsedContent = JSON.parse(fileContent.replace(/^\uFEFF/, ""));
  } catch (error) {
    throw new InvalidArgumentError(
      `Invalid JSON: ${error instanceof Error ? error.message : String(error)}`
    );
  }
  if (!Array.isArray(parsedContent) || !parsedContent.every(isRowObject)) {
    throw new InvalidArgumentError(
      "A .json file must contain an array of row objects"
    );
  }
  return parsedContent.map((row) => JSON.stringify(row)).join("\n");
}

/**
 * Fetch all datasets from Phoenix
 */
async function fetchDatasets(
  client: PhoenixClient,
  options: { limit?: number } = {}
): Promise<Dataset[]> {
  const allDatasets: Dataset[] = [];
  let cursor: string | undefined;
  const pageLimit = options.limit || 100;

  do {
    const response = await client.GET("/v1/datasets", {
      params: {
        query: {
          cursor,
          limit: pageLimit,
        },
      },
    });

    if (response.error || !response.data) {
      throw new Error(`Failed to fetch datasets: ${response.error}`);
    }

    allDatasets.push(...response.data.data);
    cursor = response.data.next_cursor || undefined;

    if (options.limit && allDatasets.length >= options.limit) {
      break;
    }
  } while (cursor);

  return allDatasets;
}

/**
 * Fetch dataset examples from Phoenix
 */
async function fetchDatasetExamples(
  client: PhoenixClient,
  datasetId: string,
  options: {
    versionId?: string;
    splits?: string[];
  } = {}
): Promise<DatasetExamplesData> {
  const response = await client.GET("/v1/datasets/{id}/examples", {
    params: {
      path: {
        id: datasetId,
      },
      query: {
        version_id: options.versionId,
        split: options.splits,
      },
    },
  });

  if (response.error || !response.data) {
    throw new Error(`Failed to fetch dataset examples: ${response.error}`);
  }

  return response.data.data;
}

/**
 * Fetch dataset metadata for display name
 */
async function fetchDatasetName(
  client: PhoenixClient,
  datasetId: string
): Promise<string | undefined> {
  try {
    const response = await client.GET("/v1/datasets/{id}", {
      params: {
        path: {
          id: datasetId,
        },
      },
    });

    if (response.error || !response.data) {
      return undefined;
    }

    return response.data.data.name;
  } catch {
    return undefined;
  }
}

/**
 * Dataset command handler
 */
async function datasetHandler(
  datasetIdentifier: string,
  options: DatasetGetOptions
): Promise<void> {
  try {
    const userSpecifiedFormat =
      process.argv.includes("--format") ||
      process.argv.some((arg) => arg.startsWith("--format="));

    // Resolve configuration
    const config = resolveConfig({
      cliOptions: {
        endpoint: options.endpoint,
        apiKey: options.apiKey,
      },
    });

    // Validate that we have endpoint
    const validation = validateConfig({ config, projectRequired: false });
    if (!validation.valid) {
      writeError({
        message: getConfigErrorMessage({ errors: validation.errors }),
      });
      process.exit(ExitCode.INVALID_ARGUMENT);
    }

    // Create client
    const client = createPhoenixClient({ config });

    writeProgress({
      message: `Resolving dataset: ${datasetIdentifier}`,
      noProgress: !options.progress,
    });

    // Resolve dataset ID
    const datasetId = await resolveDatasetId({
      client,
      datasetIdentifier,
    });

    // Fetch dataset name for display
    const datasetName = await fetchDatasetName(client, datasetId);

    writeProgress({
      message: `Fetching examples from dataset ${datasetName || datasetId}...`,
      noProgress: !options.progress,
    });

    // Fetch dataset examples
    const data = await fetchDatasetExamples(client, datasetId, {
      versionId: options.version,
      splits: options.split,
    });

    writeProgress({
      message: `Found ${data.examples.length} example(s)`,
      noProgress: !options.progress,
    });

    // Determine output format
    const outputFormat: OutputFormat = options.file
      ? "json"
      : options.format || "pretty";

    if (options.file && userSpecifiedFormat && options.format !== "json") {
      writeError({
        message: `Warning: --format is ignored when writing to a file; writing JSON to ${options.file}`,
      });
    }

    // Format output
    const output = formatDatasetExamplesOutput({
      data,
      datasetName,
      format: outputFormat,
    });

    if (options.file) {
      fs.writeFileSync(options.file, output, "utf-8");
      writeProgress({
        message: `Wrote dataset to ${options.file}`,
        noProgress: !options.progress,
      });
    } else {
      writeOutput({ message: output });
    }
  } catch (error) {
    writeError({
      message: `Error fetching dataset: ${error instanceof Error ? error.message : String(error)}`,
    });
    process.exit(getExitCodeForError(error));
  }
}

/**
 * Handler for `dataset list`
 */
async function datasetListHandler(options: DatasetListOptions): Promise<void> {
  try {
    const config = resolveConfig({
      cliOptions: {
        endpoint: options.endpoint,
        apiKey: options.apiKey,
      },
    });

    const validation = validateConfig({ config, projectRequired: false });
    if (!validation.valid) {
      writeError({
        message: getConfigErrorMessage({ errors: validation.errors }),
      });
      process.exit(ExitCode.INVALID_ARGUMENT);
    }

    const client = createPhoenixClient({ config });

    writeProgress({
      message: "Fetching datasets...",
      noProgress: !options.progress,
    });

    const datasets = await fetchDatasets(client, {
      limit: options.limit,
    });

    writeProgress({
      message: `Found ${datasets.length} dataset(s)`,
      noProgress: !options.progress,
    });

    const output = formatDatasetsOutput({
      datasets,
      format: options.format,
    });
    writeOutput({ message: output });
  } catch (error) {
    writeError({
      message: `Error fetching datasets: ${error instanceof Error ? error.message : String(error)}`,
    });
    process.exit(getExitCodeForError(error));
  }
}

/**
 * Handler for `dataset delete`
 */
async function datasetDeleteHandler(
  datasetIdentifier: string,
  options: DeleteOptions
): Promise<void> {
  try {
    assertDeletesEnabled();

    const config = resolveConfig({
      cliOptions: {
        endpoint: options.endpoint,
        apiKey: options.apiKey,
      },
    });

    const validation = validateConfig({ config, projectRequired: false });
    if (!validation.valid) {
      writeError({
        message: getConfigErrorMessage({ errors: validation.errors }),
      });
      process.exit(ExitCode.INVALID_ARGUMENT);
    }

    const client = createPhoenixClient({ config });

    writeProgress({
      message: `Resolving dataset: ${datasetIdentifier}`,
      noProgress: !options.progress,
    });

    const datasetId = await resolveDatasetId({ client, datasetIdentifier });

    await confirmOrExit({
      message: `Delete dataset ${datasetIdentifier}? This cannot be undone.`,
      yes: options.yes,
    });

    const response = await client.DELETE("/v1/datasets/{id}", {
      params: {
        path: {
          id: datasetId,
        },
      },
    });

    if (response.error) {
      throw new Error(`Failed to delete dataset: ${response.error}`);
    }

    writeProgress({
      message: `Deleted dataset ${datasetIdentifier}`,
      noProgress: !options.progress,
    });
  } catch (error) {
    writeError({
      message: `Error deleting dataset: ${error instanceof Error ? error.message : String(error)}`,
    });
    process.exit(getExitCodeForError(error));
  }
}

/**
 * Handler for `dataset upload`
 */
async function datasetUploadHandler(
  filePath: string,
  options: DatasetUploadOptions
): Promise<void> {
  try {
    const config = resolveConfig({
      cliOptions: {
        endpoint: options.endpoint,
        apiKey: options.apiKey,
      },
    });

    const validation = validateConfig({ config, projectRequired: false });
    if (!validation.valid) {
      writeError({
        message: getConfigErrorMessage({ errors: validation.errors }),
      });
      process.exit(ExitCode.INVALID_ARGUMENT);
    }

    const fileStats = fs.statSync(filePath, { throwIfNoEntry: false });
    if (!fileStats) {
      throw new InvalidArgumentError(`File not found: ${filePath}`);
    }
    if (!fileStats.isFile()) {
      throw new InvalidArgumentError(`Not a file: ${filePath}`);
    }
    const fileFormat = resolveUploadFileFormat({
      filePath,
      formatOverride: options.fileFormat,
    });
    const fileBytes = fs.readFileSync(filePath);
    const isArrowIpcFile =
      fileFormat === "arrow" &&
      fileBytes.subarray(0, ARROW_IPC_FILE_MAGIC.length).toString("latin1") ===
        ARROW_IPC_FILE_MAGIC;
    if (isArrowIpcFile) {
      throw new InvalidArgumentError(
        `${filePath} is in the Arrow IPC file format, but the upload endpoint reads the Arrow IPC stream format. Write it with pyarrow.ipc.new_stream instead.`
      );
    }
    const fileContent =
      fileFormat === "json"
        ? convertJsonRowsToJsonl(fileBytes.toString("utf-8"))
        : fileBytes;

    const formData = new FormData();
    formData.append(
      "file",
      new Blob([fileContent], { type: UPLOAD_CONTENT_TYPES[fileFormat] }),
      path.basename(filePath)
    );
    formData.append("action", options.append ? "append" : "create");
    formData.append("name", options.name);
    options.inputKeys.forEach((key) => formData.append("input_keys[]", key));
    options.outputKeys?.forEach((key) => formData.append("output_keys[]", key));
    options.metadataKeys?.forEach((key) =>
      formData.append("metadata_keys[]", key)
    );
    if (options.splitKey) {
      formData.append("split_key", options.splitKey);
    }

    const client = createPhoenixClient({ config });

    writeProgress({
      message: `Uploading ${filePath} (${fileFormat}) to dataset ${options.name}...`,
      noProgress: !options.progress,
    });

    const response = await client
      .POST("/v1/datasets/upload", {
        params: { query: { sync: true } },
        // FormData doesn't match the typed body shape; bodySerializer is what actually gets sent.
        body: formData as never,
        bodySerializer: () => formData,
      })
      .catch(async (error: unknown) => {
        if (!(error instanceof HttpError)) {
          throw error;
        }
        // The server explains rejected uploads (a taken name, an unknown
        // column) in a plain-text body; the status line alone isn't actionable.
        const errorDetail =
          (await error.response.text().catch(() => "")).trim() || error.message;
        if (error.status === 401 || error.status === 403) {
          throw new AuthRequiredError(errorDetail);
        }
        if (error.status === 409) {
          throw new InvalidArgumentError(
            `${errorDetail} Pass --append to add to it, or choose another --name.`
          );
        }
        if (error.status === 422) {
          throw new InvalidArgumentError(errorDetail);
        }
        throw new Error(errorDetail);
      });

    if (!response.data) {
      throw new Error("The server returned no upload result");
    }

    writeOutput({
      message: formatDatasetUploadOutput({
        data: response.data.data,
        datasetName: options.name,
        format: options.format,
      }),
    });
  } catch (error) {
    writeError({
      message: `Error uploading dataset: ${error instanceof Error ? error.message : String(error)}`,
    });
    process.exit(getExitCodeForError(error));
  }
}

/**
 * Create the `dataset get` command
 */
export function createDatasetGetCommand(): Command {
  return new Command("get")
    .description("Fetch examples from a dataset")
    .argument("<dataset-identifier>", "Dataset name or ID")
    .option("--endpoint <url>", "Phoenix API endpoint")
    .option("--api-key <key>", "Phoenix API key for authentication")
    .option(
      "--format <format>",
      "Output format: pretty, json, or raw",
      "pretty"
    )
    .option("--no-progress", "Disable progress indicators")
    .option("--file <path>", "Save output to file instead of stdout")
    .option(
      "--split <name>",
      "Filter by split name (can be used multiple times)",
      collectString,
      []
    )
    .option(
      "--version <id>",
      "Fetch from a specific dataset version (discover IDs with px dataset-versions list)"
    )
    .action(datasetHandler);
}

export function createDatasetListCommand(): Command {
  return new Command("list")
    .description("List all available Phoenix datasets")
    .option("--endpoint <url>", "Phoenix API endpoint")
    .option("--api-key <key>", "Phoenix API key for authentication")
    .option(
      "--format <format>",
      "Output format: pretty, json, or raw",
      "pretty"
    )
    .option("--no-progress", "Disable progress indicators")
    .option("--limit <number>", "Maximum number of datasets to fetch", parseInt)
    .action(datasetListHandler);
}

/**
 * Create the `dataset delete` command
 */
export function createDatasetDeleteCommand(): Command {
  return new Command("delete")
    .description("Delete a dataset")
    .argument("<dataset-identifier>", "Dataset name or ID")
    .option("--endpoint <url>", "Phoenix API endpoint")
    .option("--api-key <key>", "Phoenix API key for authentication")
    .option("-y, --yes", "Skip confirmation prompt")
    .option("--no-progress", "Disable progress indicators")
    .action(datasetDeleteHandler);
}

/**
 * Create the `dataset upload` command
 */
export function createDatasetUploadCommand(): Command {
  return new Command("upload")
    .description(
      "Create a dataset, or append to one, from a CSV, JSONL, JSON, or Arrow file"
    )
    .argument("<file>", "Path to the file to upload")
    .requiredOption("--name <name>", "Dataset name")
    .requiredOption(
      "--input-keys <columns...>",
      "Columns to use as example inputs"
    )
    .option("--output-keys <columns...>", "Columns to use as example outputs")
    .option(
      "--metadata-keys <columns...>",
      "Columns to store as example metadata"
    )
    .option(
      "--split-key <column>",
      "Column whose values are split names or JSON lists of split names"
    )
    .option(
      "--file-format <format>",
      "File format: csv, jsonl, json, or arrow (default: detected from the extension)"
    )
    .option(
      "--append",
      "Add to the dataset named by --name, creating it if needed"
    )
    .option("--endpoint <url>", "Phoenix API endpoint")
    .option("--api-key <key>", "Phoenix API key for authentication")
    .option(
      "--format <format>",
      "Output format: pretty, json, or raw",
      "pretty"
    )
    .option("--no-progress", "Disable progress indicators")
    .addHelpText(
      "after",
      [
        "",
        "A .json file must contain an array of row objects. An .arrow file must be",
        "an Arrow IPC stream.",
        "",
        "Examples:",
        "  px dataset upload examples.csv --name qa --input-keys question context --output-keys answer",
        "  px dataset upload more.jsonl --name qa --input-keys question --output-keys answer --append",
        "  px dataset upload rows.txt --name qa --input-keys question --file-format jsonl --format raw --no-progress",
      ].join("\n")
    )
    .action(datasetUploadHandler);
}

/**
 * Create the `dataset` command with subcommands
 */
export function createDatasetCommand(): Command {
  const command = new Command("dataset");
  command.description("Manage Phoenix datasets");
  command.addCommand(createDatasetListCommand());
  command.addCommand(createDatasetGetCommand());
  command.addCommand(createDatasetUploadCommand());
  command.addCommand(createDatasetDeleteCommand());
  return command;
}
