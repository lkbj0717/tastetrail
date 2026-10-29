# Data and licenses

## Team software

The team's source code and documentation are available under the root [MIT license](../LICENSE), with copyright attributed to TasteTrail contributors. This license does not grant rights to third-party services, datasets, trademarks, or model weights.

## MovieLens

The runtime bootstrap downloads MovieLens 1M from the official GroupLens distribution and retains its accompanying README. Review the [dataset conditions](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt) before use and preserve the required acknowledgments and restrictions. The public repository does not redistribute raw ratings, user identifiers, or background rating matrices.

Dataset reference: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems 5(4), Article 19. [DOI](https://doi.org/10.1145/2827872).

## Qloo

Qloo keys and API responses are subject to Qloo's service and hackathon terms. The code license does not relicense Qloo data. Runtime responses and derived metadata remain in private server storage; no key is embedded in the browser or repository. Requests use allowed public entity fields, without names, email addresses, original dataset user IDs, or hidden candidate ratings.

See the [Qloo hackathon developer guide](https://docs.qloo.com/reference/qloo-llm-hackathon-developer-guide) and [official starter kit](https://github.com/qloo/qloo-hackathon-kit). Qloo affinity is population-level cultural association, not an individual preference probability.

## Model service

The recorded experiment used a third-party compatible API reporting the alias `glm-5.3-flash`. Credentials and access rights must be obtained from an authorized service. No model weights are distributed. The alias alone does not establish the exact underlying weights or stable future behavior.

## Research evidence

Only aggregate metrics, method configuration, source code, and provenance fingerprints are distributed. Private user-level evaluations and raw API traces are excluded. The fictional example request is a product demonstration and is not a claim about a named person's taste or a benchmark success.

The repository's `.gitignore` helps prevent accidental inclusion of credentials, datasets, caches, and runtime assets. It does not replace review of files staged for publication.
