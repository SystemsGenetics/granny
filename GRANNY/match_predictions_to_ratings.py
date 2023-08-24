import pandas as pd

meta_data = pd.read_csv(
    "../../../pear-color-sorting/01-input_data/Mat-1-22_PearColorData_SubsetForNhan.xlsx - Sheet1.csv"
)
ratings = pd.read_csv(
    "../../../pear-color-sorting/03-MATLAB/peel_colors.csv", header=None
)

ratings = ratings.sort_values(0, axis="index")


def add_ratings(meta_data: pd.DataFrame, ratings: pd.DataFrame):
    meta_data_with_ratings = pd.DataFrame(
        columns=list(meta_data.columns).append(["Ratings", "Scores"])
    )

    for each_pear in ratings.iloc:
        split_file_name = each_pear[0].split("_")
        rating = each_pear[1]
        score = each_pear[2]
        file_name = "_".join(split_file_name[0:2]) + ".JPG"
        try:
            pear_data = meta_data[
                meta_data["ShadeSide_ImageFile"] == file_name
            ].iloc[int(split_file_name[2]) - 1]
            pear_data["Ratings"] = rating / 2
            pear_data["Scores"] = score
            meta_data_with_ratings = meta_data_with_ratings.append(pear_data)
        except IndexError:
            pass

    return meta_data_with_ratings


pear_meta_data = add_ratings(meta_data, ratings)

pear_meta_data.to_csv(
    "../../../pear-color-sorting/03-MATLAB/Nhan's_Rating_Mat-1-22_PearColorData_SubsetForNhan.xlsx - Sheet1.csv"
)
