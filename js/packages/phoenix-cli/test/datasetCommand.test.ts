import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import type { componentsV1 } from "@arizeai/phoenix-testing";
import { HttpResponse } from "@arizeai/phoenix-testing";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createDatasetCommand } from "../src/commands/dataset";
import { ExitCode } from "../src/exitCodes";
import { http, setupMockPhoenixServer } from "./mockServer";
import { BASE_ARGS, captureCliOutput, mockProcessExit } from "./testUtils";

type Dataset = componentsV1["schemas"]["Dataset"];
type DatasetExample = componentsV1["schemas"]["DatasetExample"];
type UploadDatasetData = componentsV1["schemas"]["UploadDatasetData"];

const mock = setupMockPhoenixServer();

// A hex identifier is treated as a dataset ID directly, skipping the
// `/v1/datasets?name=...` name-resolution round-trip.
const DATASET_ID = "abc123";

const DATASET_GOLDEN: Dataset = {
  id: DATASET_ID,
  name: "golden-set",
  description: "Curated regression examples",
  metadata: {},
  created_at: "2026-06-01T00:00:00.000Z",
  updated_at: "2026-06-15T00:00:00.000Z",
  example_count: 2,
};

const DATASET_SCRATCH: Dataset = {
  id: "def456",
  name: "scratch",
  description: null,
  metadata: {},
  created_at: "2026-06-02T00:00:00.000Z",
  updated_at: "2026-06-02T00:00:00.000Z",
  example_count: 0,
};

const EXAMPLE_ONE: DatasetExample = {
  id: "RGF0YXNldEV4YW1wbGU6MQ==",
  node_id: "RGF0YXNldEV4YW1wbGU6MQ==",
  input: { question: "What is Phoenix?" },
  output: { answer: "An AI observability platform" },
  metadata: { split: "train" },
  updated_at: "2026-06-15T00:00:00.000Z",
};

const EXAMPLE_TWO: DatasetExample = {
  id: "RGF0YXNldEV4YW1wbGU6Mg==",
  node_id: "RGF0YXNldEV4YW1wbGU6Mg==",
  input: { question: "What is a span?" },
  output: { answer: "A unit of work in a trace" },
  metadata: { split: "train" },
  updated_at: "2026-06-15T00:00:00.000Z",
};

afterEach(() => {
  vi.restoreAllMocks();
});

describe("dataset list", () => {
  it("outputs pinned datasets in raw mode and propagates --limit", async () => {
    let capturedQuery: URLSearchParams | undefined;
    mock.server.use(
      http.get("/v1/datasets", ({ request, response }) => {
        capturedQuery = new URL(request.url).searchParams;
        return response(200).json({
          data: [DATASET_GOLDEN, DATASET_SCRATCH],
          next_cursor: null,
        });
      })
    );
    const io = captureCliOutput();

    await createDatasetCommand().parseAsync(
      ["list", "--format", "raw", "--limit", "50", ...BASE_ARGS],
      { from: "user" }
    );

    expect(capturedQuery?.get("limit")).toBe("50");
    expect(capturedQuery?.get("cursor")).toBeNull();

    const output = io.stdout.mock.calls[0]?.[0];
    const datasets = JSON.parse(String(output));
    expect(datasets).toEqual([DATASET_GOLDEN, DATASET_SCRATCH]);
  });

  it("follows next_cursor across pages", async () => {
    const capturedCursors: Array<string | null> = [];
    let page = 0;
    mock.server.use(
      http.get("/v1/datasets", ({ request, response }) => {
        capturedCursors.push(new URL(request.url).searchParams.get("cursor"));
        page += 1;
        if (page === 1) {
          return response(200).json({
            data: [DATASET_GOLDEN],
            next_cursor: "cursor-page-2",
          });
        }
        return response(200).json({
          data: [DATASET_SCRATCH],
          next_cursor: null,
        });
      })
    );
    const io = captureCliOutput();

    await createDatasetCommand().parseAsync(
      ["list", "--format", "raw", ...BASE_ARGS],
      { from: "user" }
    );

    expect(capturedCursors).toEqual([null, "cursor-page-2"]);
    const output = io.stdout.mock.calls[0]?.[0];
    const datasets = JSON.parse(String(output));
    expect(datasets.map((d: Dataset) => d.name)).toEqual([
      "golden-set",
      "scratch",
    ]);
  });

  it("exits NETWORK_ERROR when the request fails at the network level", async () => {
    mock.server.use(http.get("/v1/datasets", () => HttpResponse.error()));
    const io = captureCliOutput();
    const exitSpy = mockProcessExit();

    await expect(
      createDatasetCommand().parseAsync(
        ["list", "--format", "raw", ...BASE_ARGS],
        { from: "user" }
      )
    ).rejects.toThrow(`process.exit:${ExitCode.NETWORK_ERROR}`);

    expect(exitSpy).toHaveBeenCalledWith(ExitCode.NETWORK_ERROR);
    expect(String(io.stderr.mock.calls[0]?.[0])).toContain(
      "Error fetching datasets"
    );
  });

  it("completes end-to-end against the generated OpenAPI handlers", async () => {
    // No pinned handlers: every response comes from the schema-generated mocks.
    const io = captureCliOutput();
    const exitSpy = mockProcessExit();

    await createDatasetCommand().parseAsync(
      ["list", "--format", "raw", "--limit", "2", ...BASE_ARGS],
      { from: "user" }
    );

    expect(exitSpy).not.toHaveBeenCalled();
    const output = io.stdout.mock.calls[0]?.[0];
    const datasets = JSON.parse(String(output));
    expect(Array.isArray(datasets)).toBe(true);
    expect(datasets.length).toBeGreaterThan(0);
    expect(typeof datasets[0].id).toBe("string");
  });
});

describe("dataset get", () => {
  it("resolves a dataset name to an ID and propagates version and split", async () => {
    let capturedResolutionQuery: URLSearchParams | undefined;
    let capturedExamplesId: string | undefined;
    let capturedExamplesQuery: URLSearchParams | undefined;
    mock.server.use(
      // Name resolution: a non-hex identifier is looked up by name.
      http.get("/v1/datasets", ({ request, response }) => {
        capturedResolutionQuery = new URL(request.url).searchParams;
        return response(200).json({
          data: [DATASET_GOLDEN],
          next_cursor: null,
        });
      }),
      // Dataset metadata fetch (for the display name).
      http.get("/v1/datasets/{id}", ({ response }) =>
        response(200).json({ data: DATASET_GOLDEN })
      ),
      http.get(
        "/v1/datasets/{id}/examples",
        ({ params, request, response }) => {
          capturedExamplesId = params.id;
          capturedExamplesQuery = new URL(request.url).searchParams;
          return response(200).json({
            data: {
              dataset_id: DATASET_ID,
              version_id: "version-007",
              filtered_splits: ["train"],
              examples: [EXAMPLE_ONE, EXAMPLE_TWO],
            },
          });
        }
      )
    );
    const io = captureCliOutput();

    await createDatasetCommand().parseAsync(
      [
        "get",
        "golden-set",
        "--format",
        "raw",
        "--version",
        "version-007",
        "--split",
        "train",
        ...BASE_ARGS,
      ],
      { from: "user" }
    );

    expect(capturedResolutionQuery?.get("name")).toBe("golden-set");
    expect(capturedResolutionQuery?.get("limit")).toBe("1");
    expect(capturedExamplesId).toBe(DATASET_ID);
    expect(capturedExamplesQuery?.get("version_id")).toBe("version-007");
    expect(capturedExamplesQuery?.getAll("split")).toEqual(["train"]);

    const output = io.stdout.mock.calls[0]?.[0];
    const parsed = JSON.parse(String(output));
    expect(parsed).toEqual({
      dataset_id: DATASET_ID,
      version_id: "version-007",
      filtered_splits: ["train"],
      examples: [EXAMPLE_ONE, EXAMPLE_TWO],
    });
  });

  it("skips name resolution when the identifier is a hex ID", async () => {
    let resolutionCalls = 0;
    mock.server.use(
      http.get("/v1/datasets", ({ response }) => {
        resolutionCalls += 1;
        return response(200).json({ data: [], next_cursor: null });
      }),
      http.get("/v1/datasets/{id}", ({ response }) =>
        response(200).json({ data: DATASET_GOLDEN })
      ),
      http.get("/v1/datasets/{id}/examples", ({ params, response }) =>
        response(200).json({
          data: {
            dataset_id: params.id,
            version_id: "version-latest",
            examples: [EXAMPLE_ONE],
          },
        })
      )
    );
    const io = captureCliOutput();

    await createDatasetCommand().parseAsync(
      ["get", DATASET_ID, "--format", "raw", ...BASE_ARGS],
      { from: "user" }
    );

    expect(resolutionCalls).toBe(0);
    const output = io.stdout.mock.calls[0]?.[0];
    const parsed = JSON.parse(String(output));
    expect(parsed.dataset_id).toBe(DATASET_ID);
    expect(parsed.examples).toEqual([EXAMPLE_ONE]);
  });

  it("exits FAILURE with an error on stderr when the name matches no dataset", async () => {
    mock.server.use(
      http.get("/v1/datasets", ({ response }) =>
        response(200).json({ data: [], next_cursor: null })
      )
    );
    const io = captureCliOutput();
    const exitSpy = mockProcessExit();

    await expect(
      createDatasetCommand().parseAsync(
        ["get", "no-such-dataset", "--format", "raw", ...BASE_ARGS],
        { from: "user" }
      )
    ).rejects.toThrow(`process.exit:${ExitCode.FAILURE}`);

    expect(exitSpy).toHaveBeenCalledWith(ExitCode.FAILURE);
    const stderrCall = String(io.stderr.mock.calls[0]?.[0]);
    expect(stderrCall).toContain("Error fetching dataset");
    expect(stderrCall).toContain('Dataset not found: "no-such-dataset"');
  });

  it("exits NETWORK_ERROR when the examples request fails at the network level", async () => {
    mock.server.use(
      // The display-name fetch swallows its own errors, so only the examples
      // request needs to fail to exercise the network error path.
      http.get("/v1/datasets/{id}/examples", () => HttpResponse.error())
    );
    const io = captureCliOutput();
    const exitSpy = mockProcessExit();

    await expect(
      createDatasetCommand().parseAsync(
        ["get", DATASET_ID, "--format", "raw", ...BASE_ARGS],
        { from: "user" }
      )
    ).rejects.toThrow(`process.exit:${ExitCode.NETWORK_ERROR}`);

    expect(exitSpy).toHaveBeenCalledWith(ExitCode.NETWORK_ERROR);
    expect(String(io.stderr.mock.calls[0]?.[0])).toContain(
      "Error fetching dataset"
    );
  });
});

describe("dataset upload", () => {
  const UPLOAD_RESULT: UploadDatasetData = {
    dataset_id: DATASET_ID,
    version_id: "RGF0YXNldFZlcnNpb246MQ==",
    num_created_examples: 2,
    num_updated_examples: 0,
    num_deleted_examples: 0,
  };

  let uploadDirectory: string;

  beforeEach(() => {
    uploadDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "px-upload-"));
  });

  afterEach(() => {
    fs.rmSync(uploadDirectory, { recursive: true, force: true });
  });

  function writeUploadFile({
    fileName,
    content,
  }: {
    fileName: string;
    content: string | Buffer;
  }): string {
    const filePath = path.join(uploadDirectory, fileName);
    fs.writeFileSync(filePath, content);
    return filePath;
  }

  function captureUploadRequest(): {
    getFormData: () => FormData | undefined;
    getUploadedFile: () => File;
    getQuery: () => URLSearchParams | undefined;
  } {
    let capturedFormData: FormData | undefined;
    let capturedQuery: URLSearchParams | undefined;
    mock.server.use(
      http.post("/v1/datasets/upload", async ({ request, response }) => {
        capturedQuery = new URL(request.url).searchParams;
        capturedFormData = await request.formData();
        return response(200).json({ data: UPLOAD_RESULT });
      })
    );
    return {
      getFormData: () => capturedFormData,
      getUploadedFile: () => {
        const uploadedFile = capturedFormData?.get("file");
        if (!(uploadedFile instanceof File)) {
          throw new Error("No file was uploaded");
        }
        return uploadedFile;
      },
      getQuery: () => capturedQuery,
    };
  }

  it("uploads a CSV with its column keys and prints the result", async () => {
    const csvContent =
      "question,context,answer,source,split\nWhat?,ctx,That,docs,train\n";
    const filePath = writeUploadFile({
      fileName: "examples.csv",
      content: csvContent,
    });
    const capture = captureUploadRequest();
    const io = captureCliOutput();

    await createDatasetCommand().parseAsync(
      [
        "upload",
        filePath,
        "--name",
        "qa",
        "--input-keys",
        "question",
        "context",
        "--output-keys",
        "answer",
        "--metadata-keys",
        "source",
        "--split-key",
        "split",
        "--format",
        "raw",
        ...BASE_ARGS,
      ],
      { from: "user" }
    );

    const formData = capture.getFormData();
    const file = capture.getUploadedFile();
    expect(file.type).toBe("text/csv");
    expect(file.name).toBe("examples.csv");
    expect(await file.text()).toBe(csvContent);
    expect(formData?.get("action")).toBe("create");
    expect(formData?.get("name")).toBe("qa");
    expect(formData?.getAll("input_keys[]")).toEqual(["question", "context"]);
    expect(formData?.getAll("output_keys[]")).toEqual(["answer"]);
    expect(formData?.getAll("metadata_keys[]")).toEqual(["source"]);
    expect(formData?.get("split_key")).toBe("split");
    expect(capture.getQuery()?.get("sync")).toBe("true");
    expect(JSON.parse(String(io.stdout.mock.calls[0]?.[0]))).toEqual(
      UPLOAD_RESULT
    );
  });

  it("converts a JSON array of rows to JSONL, appends with --append, and prints indented JSON", async () => {
    const filePath = writeUploadFile({
      fileName: "rows.json",
      content:
        "\uFEFF" +
        JSON.stringify([
          { question: "What?", answer: "That" },
          { question: "Why?", answer: "Because" },
        ]),
    });
    const capture = captureUploadRequest();
    const io = captureCliOutput();

    await createDatasetCommand().parseAsync(
      [
        "upload",
        filePath,
        "--name",
        "qa",
        "--input-keys",
        "question",
        "--append",
        "--format",
        "json",
        ...BASE_ARGS,
      ],
      { from: "user" }
    );

    const file = capture.getUploadedFile();
    expect(file.type).toBe("application/jsonl");
    expect(await file.text()).toBe(
      '{"question":"What?","answer":"That"}\n{"question":"Why?","answer":"Because"}'
    );
    expect(capture.getFormData()?.get("action")).toBe("append");
    expect(capture.getFormData()?.getAll("output_keys[]")).toEqual([]);
    expect(String(io.stdout.mock.calls[0]?.[0])).toBe(
      JSON.stringify(UPLOAD_RESULT, null, 2)
    );
  });

  it("detects an Arrow file from its extension and sends its bytes unchanged", async () => {
    const arrowBytes = Buffer.from([0xff, 0xff, 0xff, 0xff, 0x00, 0x01]);
    const filePath = writeUploadFile({
      fileName: "examples.arrow",
      content: arrowBytes,
    });
    const capture = captureUploadRequest();
    captureCliOutput();

    await createDatasetCommand().parseAsync(
      [
        "upload",
        filePath,
        "--name",
        "qa",
        "--input-keys",
        "question",
        ...BASE_ARGS,
      ],
      { from: "user" }
    );

    const file = capture.getUploadedFile();
    expect(file.type).toBe("application/x-pandas-pyarrow");
    expect(Buffer.from(await file.arrayBuffer())).toEqual(arrowBytes);
  });

  it("uses --file-format, case-insensitively, over the file extension", async () => {
    const filePath = writeUploadFile({
      fileName: "rows.txt",
      content: '{"question":"What?"}\n',
    });
    const capture = captureUploadRequest();
    captureCliOutput();

    await createDatasetCommand().parseAsync(
      [
        "upload",
        filePath,
        "--name",
        "qa",
        "--input-keys",
        "question",
        "--file-format",
        "JSONL",
        ...BASE_ARGS,
      ],
      { from: "user" }
    );

    expect(capture.getUploadedFile().type).toBe("application/jsonl");
  });

  it("prints the dataset, version, and created count in pretty mode", async () => {
    const filePath = writeUploadFile({
      fileName: "examples.csv",
      content: "question\nWhat?\n",
    });
    captureUploadRequest();
    const io = captureCliOutput();

    await createDatasetCommand().parseAsync(
      [
        "upload",
        filePath,
        "--name",
        "qa",
        "--input-keys",
        "question",
        ...BASE_ARGS,
      ],
      { from: "user" }
    );

    expect(String(io.stdout.mock.calls[0]?.[0])).toBe(
      [
        `Dataset: qa (${DATASET_ID})`,
        "Version: RGF0YXNldFZlcnNpb246MQ==",
        "Examples created: 2",
        "Examples updated: 0",
        "Examples deleted: 0",
      ].join("\n")
    );
  });

  it.each([
    {
      case: "an extension it cannot detect",
      fileName: "examples.txt",
      content: "question\nWhat?\n",
      extraArgs: [],
      expectedError: "Cannot detect the format",
    },
    {
      case: "an unsupported --file-format",
      fileName: "examples.csv",
      content: "question\nWhat?\n",
      extraArgs: ["--file-format", "xlsx"],
      expectedError: 'Unsupported --file-format "xlsx"',
    },
    {
      case: "a .json file that is not an array of objects",
      fileName: "rows.json",
      content: '{"question":"What?"}',
      extraArgs: [],
      expectedError: "must contain an array of row objects",
    },
    {
      case: "a .json file that is not valid JSON",
      fileName: "rows.json",
      content: "[{",
      extraArgs: [],
      expectedError: "Invalid JSON",
    },
    {
      case: "an Arrow IPC file instead of a stream",
      fileName: "examples.arrow",
      content: "ARROW1\u0000\u0000",
      extraArgs: [],
      expectedError: "Write it with pyarrow.ipc.new_stream",
    },
  ])(
    "exits INVALID_ARGUMENT without uploading for $case",
    async ({ fileName, content, extraArgs, expectedError }) => {
      const filePath = writeUploadFile({ fileName, content });
      let isUploadRequested = false;
      mock.server.use(
        http.post("/v1/datasets/upload", ({ response }) => {
          isUploadRequested = true;
          return response(200).json({ data: UPLOAD_RESULT });
        })
      );
      const io = captureCliOutput();
      const exitSpy = mockProcessExit();

      await expect(
        createDatasetCommand().parseAsync(
          [
            "upload",
            filePath,
            "--name",
            "qa",
            "--input-keys",
            "question",
            ...extraArgs,
            ...BASE_ARGS,
          ],
          { from: "user" }
        )
      ).rejects.toThrow(`process.exit:${ExitCode.INVALID_ARGUMENT}`);

      expect(exitSpy).toHaveBeenCalledWith(ExitCode.INVALID_ARGUMENT);
      expect(isUploadRequested).toBe(false);
      expect(String(io.stderr.mock.calls[0]?.[0])).toContain(expectedError);
    }
  );

  it.each([
    {
      case: "does not exist",
      getFilePath: () => path.join(uploadDirectory, "missing.csv"),
      expectedError: "File not found",
    },
    {
      case: "is a directory",
      getFilePath: () => uploadDirectory,
      expectedError: "Not a file",
    },
  ])(
    "exits INVALID_ARGUMENT when the path $case",
    async ({ getFilePath, expectedError }) => {
      const io = captureCliOutput();
      const exitSpy = mockProcessExit();

      await expect(
        createDatasetCommand().parseAsync(
          [
            "upload",
            getFilePath(),
            "--name",
            "qa",
            "--input-keys",
            "question",
            ...BASE_ARGS,
          ],
          { from: "user" }
        )
      ).rejects.toThrow(`process.exit:${ExitCode.INVALID_ARGUMENT}`);

      expect(exitSpy).toHaveBeenCalledWith(ExitCode.INVALID_ARGUMENT);
      expect(String(io.stderr.mock.calls[0]?.[0])).toContain(expectedError);
    }
  );

  it.each([
    {
      status: 409,
      body: 'A dataset named "qa" already exists.',
      expectedExitCode: ExitCode.INVALID_ARGUMENT,
      expectedError:
        'A dataset named "qa" already exists. Pass --append to add to it, or choose another --name.',
    },
    {
      status: 422,
      body: "input keys not found in column headers: frozenset({'missing'})",
      expectedExitCode: ExitCode.INVALID_ARGUMENT,
      expectedError: "input keys not found in column headers",
    },
    {
      status: 401,
      body: "Invalid API key",
      expectedExitCode: ExitCode.AUTH_REQUIRED,
      expectedError: "Invalid API key",
    },
    {
      status: 500,
      body: "",
      expectedExitCode: ExitCode.FAILURE,
      expectedError: "500 Internal Server Error",
    },
  ])(
    "reports the server's explanation for a $status response",
    async ({ status, body, expectedExitCode, expectedError }) => {
      const filePath = writeUploadFile({
        fileName: "examples.csv",
        content: "question\nWhat?\n",
      });
      mock.server.use(
        http.post(
          "/v1/datasets/upload",
          () =>
            new HttpResponse(body, {
              status,
              headers: { "content-type": "text/plain; charset=utf-8" },
            })
        )
      );
      const io = captureCliOutput();
      const exitSpy = mockProcessExit();

      await expect(
        createDatasetCommand().parseAsync(
          [
            "upload",
            filePath,
            "--name",
            "qa",
            "--input-keys",
            "question",
            ...BASE_ARGS,
          ],
          { from: "user" }
        )
      ).rejects.toThrow(`process.exit:${expectedExitCode}`);

      expect(exitSpy).toHaveBeenCalledWith(expectedExitCode);
      expect(String(io.stderr.mock.calls[0]?.[0])).toContain(expectedError);
    }
  );

  it("exits NETWORK_ERROR when the upload fails at the network level", async () => {
    const filePath = writeUploadFile({
      fileName: "examples.csv",
      content: "question\nWhat?\n",
    });
    mock.server.use(
      http.post("/v1/datasets/upload", () => HttpResponse.error())
    );
    const io = captureCliOutput();
    const exitSpy = mockProcessExit();

    await expect(
      createDatasetCommand().parseAsync(
        [
          "upload",
          filePath,
          "--name",
          "qa",
          "--input-keys",
          "question",
          ...BASE_ARGS,
        ],
        { from: "user" }
      )
    ).rejects.toThrow(`process.exit:${ExitCode.NETWORK_ERROR}`);

    expect(exitSpy).toHaveBeenCalledWith(ExitCode.NETWORK_ERROR);
    expect(String(io.stderr.mock.calls[0]?.[0])).toContain(
      "Error uploading dataset"
    );
  });

  it("completes end-to-end against the generated OpenAPI handlers", async () => {
    // No pinned handlers: every response comes from the schema-generated mocks.
    const filePath = writeUploadFile({
      fileName: "examples.csv",
      content: "question\nWhat?\n",
    });
    const io = captureCliOutput();
    const exitSpy = mockProcessExit();

    await createDatasetCommand().parseAsync(
      [
        "upload",
        filePath,
        "--name",
        "qa",
        "--input-keys",
        "question",
        "--format",
        "raw",
        ...BASE_ARGS,
      ],
      { from: "user" }
    );

    expect(exitSpy).not.toHaveBeenCalled();
    const result = JSON.parse(String(io.stdout.mock.calls[0]?.[0]));
    expect(typeof result.dataset_id).toBe("string");
    expect(typeof result.num_created_examples).toBe("number");
  });
});
