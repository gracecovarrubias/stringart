# String Art With AI

String Art With AI is a modified version of grvlbit's string-art generator with the goal of using reinforcement learning to train a model for creating string art.

![Mona Lisa stringart](stringart/demo/result_ml.png "Mona Lisa stringart")

This project has been inspired by the magnificant works of [Petros
Vrellis](http://artof01.com/vrellis/works/knit.html).

## Algorithm

The idea is to crop a picture to a given shape (e.g. circle) and then
place a number of nails evenly spaced around that. Then based on a random nail
the algorithm starts calculating the best route (that is the route with the highest
darkness) to the next nail. This procedure is continued until a maximum number
of iterations is reached or the simulated picture matches the input.

## Usage

To run the project, we recommend using the devcontainer.

To open the project in the devcontainer in vscode, use `ctrl + shift + p` to access the avaiolable commands and select `reopen in devcontainer`.

## Notes

This is a relatively naive implementation. The problem scales with the number
of nails, iterations as well as the resolution of the input.

* Use cropped images roughly 300x300px
* Works best with pictures with high contrast
* Works best on portraits
