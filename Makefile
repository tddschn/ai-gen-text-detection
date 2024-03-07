ipynb-to-py:
	fd -e ipynb -x pandoc {} -o {.}.py	